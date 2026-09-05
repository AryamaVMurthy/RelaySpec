from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from typing import Any


MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


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
            raise ValueError(f"incomplete paired benchmark for {key}; missing={missing}")

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
            by_method[method]["output_hash"]
            == by_method[native_method]["output_hash"]
            for by_method in grouped.values()
        )
        summary["exact_sequence_matches"] = exact
        summary["exact_sequence_match_rate"] = exact / len(grouped)

    return {
        "paired_requests": len(grouped),
        "methods": method_summary,
    }
