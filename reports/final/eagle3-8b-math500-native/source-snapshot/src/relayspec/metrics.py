from __future__ import annotations

import math
import random
import statistics
from collections.abc import Iterable, Sequence
from typing import Any


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of no values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_mismatch_audit(
    rows: Iterable[dict[str, Any]],
    *,
    reference_method: str,
    candidate_method: str,
    context_chars: int = 160,
) -> dict[str, Any]:
    """Retain every paired text mismatch and its independently scored outcome."""
    if context_chars <= 0:
        raise ValueError("context_chars must be positive")
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    selected_methods = {reference_method, candidate_method}
    for row in rows:
        method = str(row["method"])
        if method not in selected_methods:
            continue
        key = (str(row["problem_id"]), int(row.get("repetition", 0)))
        by_method = grouped.setdefault(key, {})
        if method in by_method:
            raise ValueError(f"duplicate paired row: {key} {method}")
        by_method[method] = row
    if not grouped:
        raise ValueError("no paired rows found")

    mismatches: list[dict[str, Any]] = []
    for (problem_id, repetition), by_method in sorted(grouped.items()):
        if set(by_method) != selected_methods:
            raise ValueError(f"incomplete pair: {(problem_id, repetition)}")
        reference = by_method[reference_method]
        candidate = by_method[candidate_method]
        reference_text = str(reference.get("completion", ""))
        candidate_text = str(candidate.get("completion", ""))
        same = (
            reference.get("output_hash") == candidate.get("output_hash")
            if "output_hash" in reference and "output_hash" in candidate
            else reference_text == candidate_text
        )
        if same:
            continue
        common_prefix = 0
        for reference_char, candidate_char in zip(reference_text, candidate_text):
            if reference_char != candidate_char:
                break
            common_prefix += 1
        start = max(0, common_prefix - context_chars)
        stop = common_prefix + context_chars
        mismatches.append(
            {
                "problem_id": problem_id,
                "repetition": repetition,
                "common_prefix_chars": common_prefix,
                "reference_output_tokens": int(reference.get("output_tokens", 0)),
                "candidate_output_tokens": int(candidate.get("output_tokens", 0)),
                "reference_correct": reference.get("correct"),
                "candidate_correct": candidate.get("correct"),
                "reference_tail": reference_text[start:stop],
                "candidate_tail": candidate_text[start:stop],
            }
        )
    return {
        "reference_method": reference_method,
        "candidate_method": candidate_method,
        "paired_requests": len(grouped),
        "mismatch_count": len(mismatches),
        "match_rate": 1.0 - len(mismatches) / len(grouped),
        "mismatches": mismatches,
    }


def paired_bootstrap_summary(
    rows: Iterable[dict[str, Any]],
    *,
    methods: Sequence[str],
    reference_method: str = "native_ar",
    samples: int = 10_000,
    seed: int = 1729,
) -> dict[str, Any]:
    method_names = tuple(methods)
    if reference_method not in method_names:
        raise ValueError("reference method is missing")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["problem_id"]), int(row.get("repetition", 0)))
        grouped.setdefault(key, {})[str(row["method"])] = row
    if not grouped:
        raise ValueError("benchmark produced no paired rows")
    required = set(method_names)
    for key, by_method in grouped.items():
        if set(by_method) != required:
            raise ValueError(f"incomplete pair for {key}")

    pairs = list(grouped.values())

    has_request_time = all(
        "request_seconds" in pair[method]
        for pair in pairs
        for method in method_names
    )
    has_hashes = all(
        "output_hash" in pair[method]
        for pair in pairs
        for method in method_names
    )
    has_accuracy = all(
        pair[method].get("correct") is not None
        for pair in pairs
        for method in method_names
    )

    def totals(selected: list[dict[str, dict[str, Any]]], method: str):
        tokens = sum(int(pair[method]["output_tokens"]) for pair in selected)
        seconds = sum(float(pair[method]["decode_seconds"]) for pair in selected)
        accuracy = None
        if has_accuracy:
            accuracy = sum(bool(pair[method]["correct"]) for pair in selected) / len(
                selected
            )
        request_throughput = None
        if has_request_time:
            request_seconds = sum(
                float(pair[method]["request_seconds"]) for pair in selected
            )
            request_throughput = tokens / request_seconds
        return tokens / seconds, request_throughput, accuracy

    point_reference_tps, point_reference_request_tps, point_reference_accuracy = (
        totals(pairs, reference_method)
    )
    source_method = next(
        (
            name
            for name in (
                "optimized_source_reuse",
                "naive_source_reuse",
                "source_reuse_eagle3",
            )
            if name in method_names
        ),
        None,
    )
    point_source = totals(pairs, source_method) if source_method is not None else None
    result: dict[str, Any] = {
        "paired_requests": len(pairs),
        "reference_method": reference_method,
        "methods": {},
    }
    for method in method_names:
        throughput, request_throughput, accuracy = totals(pairs, method)
        selected_rows = [pair[method] for pair in pairs]
        method_result = {
            "requests": len(selected_rows),
            "output_tokens": sum(int(row["output_tokens"]) for row in selected_rows),
            "throughput_tokens_per_second": throughput,
            "speedup_vs_reference": throughput / point_reference_tps,
            "mean_acceptance_length": statistics.fmean(
                float(row["acceptance_length"]) for row in selected_rows
            ),
            "mean_target_calls": statistics.fmean(
                float(row.get("target_calls", 0)) for row in selected_rows
            ),
        }
        if all("acceptance_lengths" in row for row in selected_rows):
            cycle_lengths = [
                int(length)
                for row in selected_rows
                for length in row["acceptance_lengths"]
            ]
            if not cycle_lengths or min(cycle_lengths) < 1:
                raise ValueError(f"invalid acceptance cycle lengths for {method}")
            method_result["acceptance_cycles"] = len(cycle_lengths)
            method_result["acceptance_survival_by_position"] = {
                str(position): sum(
                    length >= position for length in cycle_lengths
                )
                / len(cycle_lengths)
                for position in range(1, max(cycle_lengths) + 1)
            }
        if all("input_tokens" in row for row in selected_rows):
            method_result["mean_input_tokens"] = statistics.fmean(
                int(row["input_tokens"]) for row in selected_rows
            )
        if all("draft_calls" in row for row in selected_rows):
            method_result["mean_draft_calls"] = statistics.fmean(
                int(row["draft_calls"]) for row in selected_rows
            )
        for memory_key in (
            "allocated_memory_before_bytes",
            "reserved_memory_before_bytes",
            "peak_allocated_memory_bytes",
            "peak_reserved_memory_bytes",
        ):
            if all(memory_key in row for row in selected_rows):
                method_result[memory_key] = max(
                    int(row[memory_key]) for row in selected_rows
                )
        if all("time_to_first_token_seconds" in row for row in selected_rows):
            ttft = [float(row["time_to_first_token_seconds"]) for row in selected_rows]
            method_result["ttft_p50_seconds"] = _percentile(ttft, 0.50)
            method_result["ttft_p95_seconds"] = _percentile(ttft, 0.95)
        result["methods"][method] = method_result
        if reference_method == "native_ar":
            method_result["speedup_vs_native_ar"] = method_result[
                "speedup_vs_reference"
            ]
        if all("hit_token_cap" in row for row in selected_rows):
            method_result["cap_hit_rate"] = sum(
                bool(row["hit_token_cap"]) for row in selected_rows
            ) / len(selected_rows)
        if all("profile_regions_ms" in row for row in selected_rows):
            region_names = sorted(
                {
                    region
                    for row in selected_rows
                    for region in row["profile_regions_ms"]
                }
            )
            region_totals = {
                region: sum(
                    float(row["profile_regions_ms"].get(region, 0.0))
                    for row in selected_rows
                )
                for region in region_names
            }
            total_request_ms = sum(
                float(row["request_seconds"]) * 1000 for row in selected_rows
            )
            method_result["profile_region_totals_ms"] = region_totals
            method_result["profile_region_request_shares"] = {
                region: value / total_request_ms
                for region, value in region_totals.items()
            }
            method_result["profile_accounted_fraction"] = (
                sum(region_totals.values()) / total_request_ms
            )
        if accuracy is not None and point_reference_accuracy is not None:
            method_result["accuracy"] = accuracy
            method_result["accuracy_delta_vs_reference"] = (
                accuracy - point_reference_accuracy
            )
            if reference_method == "native_ar":
                method_result["accuracy_delta_vs_native_ar"] = method_result[
                    "accuracy_delta_vs_reference"
                ]
        if point_source is not None:
            source_tps, source_request_tps, source_accuracy = point_source
            method_result["speedup_vs_source_reuse"] = throughput / source_tps
            if request_throughput is not None and source_request_tps is not None:
                method_result["end_to_end_speedup_vs_source_reuse"] = (
                    request_throughput / source_request_tps
                )
            if accuracy is not None and source_accuracy is not None:
                method_result["accuracy_delta_vs_source_reuse"] = (
                    accuracy - source_accuracy
                )
        if request_throughput is not None and point_reference_request_tps is not None:
            method_result.update(
                {
                    "end_to_end_tokens_per_second": request_throughput,
                    "end_to_end_speedup_vs_reference": (
                        request_throughput / point_reference_request_tps
                    ),
                }
            )
            if reference_method == "native_ar":
                method_result["end_to_end_speedup_vs_native_ar"] = method_result[
                    "end_to_end_speedup_vs_reference"
                ]
        if has_hashes:
            exact = sum(
                pair[method]["output_hash"]
                == pair[reference_method]["output_hash"]
                for pair in pairs
            )
            method_result["exact_sequence_match_rate_vs_reference"] = exact / len(
                pairs
            )
            if reference_method == "native_ar":
                method_result["exact_sequence_match_rate_vs_native_ar"] = (
                    method_result["exact_sequence_match_rate_vs_reference"]
                )

    generator = random.Random(seed)
    speed_samples = {method: [] for method in method_names}
    request_speed_samples = {method: [] for method in method_names}
    accuracy_delta_samples = {method: [] for method in method_names}
    source_speed_samples = {method: [] for method in method_names}
    source_request_speed_samples = {method: [] for method in method_names}
    source_accuracy_delta_samples = {method: [] for method in method_names}
    for _ in range(samples):
        resample = [pairs[generator.randrange(len(pairs))] for _ in pairs]
        reference_tps, reference_request_tps, reference_accuracy = totals(
            resample, reference_method
        )
        source_totals = (
            totals(resample, source_method) if source_method is not None else None
        )
        for method in method_names:
            throughput, request_throughput, accuracy = totals(resample, method)
            speed_samples[method].append(throughput / reference_tps)
            if request_throughput is not None and reference_request_tps is not None:
                request_speed_samples[method].append(
                    request_throughput / reference_request_tps
                )
            if accuracy is not None and reference_accuracy is not None:
                accuracy_delta_samples[method].append(accuracy - reference_accuracy)
            if source_totals is not None:
                source_tps, source_request_tps, source_accuracy = source_totals
                source_speed_samples[method].append(throughput / source_tps)
                if request_throughput is not None and source_request_tps is not None:
                    source_request_speed_samples[method].append(
                        request_throughput / source_request_tps
                    )
                if accuracy is not None and source_accuracy is not None:
                    source_accuracy_delta_samples[method].append(
                        accuracy - source_accuracy
                    )

    for method in method_names:
        summary = result["methods"][method]
        summary["speedup_vs_reference_ci95"] = [
            _percentile(speed_samples[method], 0.025),
            _percentile(speed_samples[method], 0.975),
        ]
        if reference_method == "native_ar":
            summary["speedup_vs_native_ar_ci95"] = summary[
                "speedup_vs_reference_ci95"
            ]
        if accuracy_delta_samples[method]:
            summary["accuracy_delta_vs_reference_ci95"] = [
                _percentile(accuracy_delta_samples[method], 0.025),
                _percentile(accuracy_delta_samples[method], 0.975),
            ]
            if reference_method == "native_ar":
                summary["accuracy_delta_vs_native_ar_ci95"] = summary[
                    "accuracy_delta_vs_reference_ci95"
                ]
        if request_speed_samples[method]:
            summary["end_to_end_speedup_vs_reference_ci95"] = [
                _percentile(request_speed_samples[method], 0.025),
                _percentile(request_speed_samples[method], 0.975),
            ]
            if reference_method == "native_ar":
                summary["end_to_end_speedup_vs_native_ar_ci95"] = summary[
                    "end_to_end_speedup_vs_reference_ci95"
                ]
        if source_speed_samples[method]:
            summary["speedup_vs_source_reuse_ci95"] = [
                _percentile(source_speed_samples[method], 0.025),
                _percentile(source_speed_samples[method], 0.975),
            ]
        if source_request_speed_samples[method]:
            summary["end_to_end_speedup_vs_source_reuse_ci95"] = [
                _percentile(source_request_speed_samples[method], 0.025),
                _percentile(source_request_speed_samples[method], 0.975),
            ]
        if source_accuracy_delta_samples[method]:
            summary["accuracy_delta_vs_source_reuse_ci95"] = [
                _percentile(source_accuracy_delta_samples[method], 0.025),
                _percentile(source_accuracy_delta_samples[method], 0.975),
            ]
    if has_hashes and source_method is not None:
        for method in method_names:
            exact = sum(
                pair[method]["output_hash"]
                == pair[source_method]["output_hash"]
                for pair in pairs
            )
            result["methods"][method][
                "exact_sequence_match_rate_vs_source_reuse"
            ] = exact / len(pairs)
    return result
