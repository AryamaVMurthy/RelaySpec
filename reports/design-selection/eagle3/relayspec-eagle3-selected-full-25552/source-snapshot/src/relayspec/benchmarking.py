from __future__ import annotations

import math
import random
import statistics
from collections.abc import Iterable, Sequence
from typing import Any

MATH_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


def model_load_plan(
    methods: Sequence[str],
    *,
    unload_source_trunk: bool,
) -> dict[str, bool]:
    """Return the optional model components required by a benchmark method set."""
    selected = set(methods)
    source_reuse_names = {"naive_source_reuse", "optimized_source_reuse"}
    needs_source_provider = bool(selected & source_reuse_names)
    needs_source_heads = bool(selected & {"relay_f", "relay_p"})
    if unload_source_trunk and needs_source_provider:
        raise ValueError(
            "cannot unload the source trunk while benchmarking "
            "naive_source_reuse/optimized_source_reuse"
        )
    return {
        "load_source": needs_source_provider or needs_source_heads,
        "build_source_provider": needs_source_provider,
        "load_native_target_draft": "native_target_dflash" in selected,
        "unload_source_trunk": unload_source_trunk and needs_source_heads,
    }


def format_benchmark_prompt(record: dict[str, Any]) -> str:
    benchmark = str(record["benchmark"])
    if "prompt" not in record:
        raise ValueError(f"{benchmark} record does not contain a single-turn prompt")
    prompt = str(record["prompt"])
    if benchmark in {"math500", "gsm8k"}:
        return prompt + MATH_SUFFIX
    return prompt


def benchmark_turns(record: dict[str, Any]) -> tuple[str, ...]:
    if "turns" in record:
        turns = tuple(str(value) for value in record["turns"])
        if not turns:
            raise ValueError("multi-turn benchmark record has no turns")
        return turns
    return (format_benchmark_prompt(record),)


def shard_records(
    records: Sequence[dict[str, Any]],
    *,
    world_size: int,
    rank: int,
) -> list[dict[str, Any]]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be within world_size")
    return list(records[rank::world_size])


def rotate_methods(methods: Sequence[str], index: int) -> tuple[str, ...]:
    if not methods:
        raise ValueError("at least one benchmark method is required")
    offset = index % len(methods)
    ordered = tuple(methods)
    return ordered[offset:] + ordered[:offset]


def generation_call_counts(stats: Any) -> tuple[int, int]:
    """Normalize call counters from instrumented and official DFlash outputs.

    The pinned official DFlash implementation reports one acceptance length per
    proposal/verification cycle but does not expose explicit call counters.
    RelaySpec's instrumented generators do expose them, so prefer those values
    and use the cycle count only as the official-baseline fallback.
    """
    cycles = len(stats.acceptance_lengths)
    target_calls = int(getattr(stats, "target_calls", cycles))
    draft_calls = int(getattr(stats, "draft_calls", cycles))
    return target_calls, draft_calls


def optional_generation_stats(stats: Any) -> dict[str, Any]:
    """Return family/provider counters when an instrumented generator exposes them."""
    result: dict[str, Any] = {}
    for name in (
        "source_executed_tokens",
        "proposal_lengths",
        "accepted_draft_lengths",
        "verify_count",
    ):
        if hasattr(stats, name):
            value = getattr(stats, name)
            result[name] = (
                list(value) if isinstance(value, (list, tuple)) else int(value)
            )
    return result


def generation_timings(stats: Any, *, request_seconds: float) -> tuple[float, float]:
    """Normalize instrumented and official APIs to TTFT and decode seconds."""
    if hasattr(stats, "time_per_output_token"):
        ttft = float(getattr(stats, "time_to_first_token", 0.0))
        decode = float(stats.time_per_output_token) * int(stats.num_output_tokens)
        return ttft, decode
    return 0.0, float(request_seconds)


def paired_bootstrap_speedup(
    rows: Iterable[dict[str, Any]],
    *,
    reference_method: str,
    candidate_method: str,
    replicates: int,
    seed: int,
    confidence_level: float = 0.95,
    time_field: str = "request_seconds",
) -> dict[str, float]:
    """Paired request-cluster bootstrap for an aggregate latency speedup.

    Each resampled unit is a paired ``(problem_id, repetition)`` request.  The
    statistic is ratio-of-sums rather than a mean of request ratios, matching
    the end-to-end latency claim made by the benchmark table.
    """
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")
    grouped: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        method = str(row["method"])
        if method not in {reference_method, candidate_method}:
            continue
        key = (str(row["problem_id"]), int(row["repetition"]))
        grouped.setdefault(key, {})[method] = float(row[time_field])
    pairs: list[tuple[float, float]] = []
    for key, values in grouped.items():
        if set(values) != {reference_method, candidate_method}:
            raise ValueError(f"incomplete bootstrap pair: {key}")
        reference = values[reference_method]
        candidate = values[candidate_method]
        if reference <= 0 or candidate <= 0:
            raise ValueError(f"non-positive timing in bootstrap pair: {key}")
        pairs.append((reference, candidate))
    if not pairs:
        raise ValueError("no paired requests found")

    def ratio(sample: Sequence[tuple[float, float]]) -> float:
        return sum(reference for reference, _ in sample) / sum(
            candidate for _, candidate in sample
        )

    rng = random.Random(seed)
    values = sorted(
        ratio([pairs[rng.randrange(len(pairs))] for _ in pairs])
        for _ in range(replicates)
    )
    tail = (1.0 - confidence_level) / 2.0
    lower_index = math.floor(tail * (replicates - 1))
    upper_index = math.ceil((1.0 - tail) * (replicates - 1))
    return {
        "estimate": ratio(pairs),
        "lower": values[lower_index],
        "upper": values[upper_index],
    }


def aggregate_rows(
    rows: Iterable[dict[str, Any]],
    *,
    methods: Sequence[str],
    native_method: str = "native_ar",
) -> dict[str, Any]:
    expected_methods = tuple(methods)
    if native_method not in expected_methods:
        raise ValueError("native reference method is missing")
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        method = str(row["method"])
        if method not in expected_methods:
            raise ValueError(f"unexpected method: {method}")
        key = (str(row["problem_id"]), int(row["repetition"]))
        by_method = grouped.setdefault(key, {})
        if method in by_method:
            raise ValueError(f"duplicate paired benchmark row: {key} {method}")
        by_method[method] = row
    if not grouped:
        raise ValueError("benchmark produced no rows")
    required = set(expected_methods)
    for key, by_method in grouped.items():
        if set(by_method) != required:
            missing = sorted(required - set(by_method))
            raise ValueError(
                f"incomplete paired benchmark for {key}; missing={missing}"
            )

    method_summary: dict[str, dict[str, Any]] = {}
    for method in expected_methods:
        selected = [by_method[method] for by_method in grouped.values()]
        output_tokens = sum(int(row["output_tokens"]) for row in selected)
        decode_seconds = sum(float(row["decode_seconds"]) for row in selected)
        if output_tokens <= 0 or decode_seconds <= 0:
            raise ValueError(f"invalid token/time totals for {method}")
        method_summary[method] = {
            "requests": len(selected),
            "output_tokens": output_tokens,
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second": output_tokens / decode_seconds,
            "mean_acceptance_length": statistics.fmean(
                float(row["acceptance_length"]) for row in selected
            ),
        }

    native_tps = method_summary[native_method]["decode_tokens_per_second"]
    for method in expected_methods:
        summary = method_summary[method]
        summary["speedup_vs_native_ar"] = (
            summary["decode_tokens_per_second"] / native_tps
        )
        if method == native_method:
            summary["exact_sequence_matches"] = len(grouped)
            summary["exact_sequence_match_rate"] = 1.0
            continue
        exact = sum(
            by_method[method]["output_hash"] == by_method[native_method]["output_hash"]
            for by_method in grouped.values()
        )
        summary["exact_sequence_matches"] = exact
        summary["exact_sequence_match_rate"] = exact / len(grouped)

    pairwise: dict[str, dict[str, Any]] = {}
    for reference_index, reference in enumerate(expected_methods):
        for candidate in expected_methods[reference_index + 1 :]:
            exact = sum(
                by_method[candidate]["output_hash"]
                == by_method[reference]["output_hash"]
                for by_method in grouped.values()
            )
            pairwise[f"{candidate}_vs_{reference}"] = {
                "reference_method": reference,
                "candidate_method": candidate,
                "throughput_speedup": (
                    method_summary[candidate]["decode_tokens_per_second"]
                    / method_summary[reference]["decode_tokens_per_second"]
                ),
                "exact_sequence_matches": exact,
                "exact_sequence_match_rate": exact / len(grouped),
            }

    return {
        "paired_requests": len(grouped),
        "methods": method_summary,
        "pairwise": pairwise,
    }
