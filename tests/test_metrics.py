from __future__ import annotations

import pytest

from relayspec.metrics import (
    paired_binary_delta_summary,
    paired_bootstrap_summary,
    paired_mismatch_audit,
)


def _row(problem: str, method: str, tokens: int, seconds: float, correct: bool):
    return {
        "problem_id": problem,
        "repetition": 0,
        "method": method,
        "output_tokens": tokens,
        "input_tokens": 7,
        "time_to_first_token_seconds": seconds / 10,
        "decode_seconds": seconds,
        "request_seconds": seconds + 0.5,
        "correct": correct,
        "output_hash": f"{problem}-{method}",
        "acceptance_length": 1.0 if method == "native_ar" else 4.0,
        "acceptance_lengths": [1, 1] if method == "native_ar" else [2, 4],
        "target_calls": tokens if method == "native_ar" else 3,
        "draft_calls": 0 if method == "native_ar" else 2,
        "allocated_memory_before_bytes": 100 if method == "native_ar" else 80,
        "peak_allocated_memory_bytes": 120 if method == "native_ar" else 95,
        "reserved_memory_before_bytes": 140 if method == "native_ar" else 110,
        "peak_reserved_memory_bytes": 160 if method == "native_ar" else 130,
    }


def test_paired_bootstrap_summary_reports_speed_and_accuracy_delta() -> None:
    rows = [
        _row("p0", "native_ar", 10, 2.0, True),
        _row("p0", "relay_p", 10, 1.0, True),
        _row("p1", "native_ar", 20, 4.0, False),
        _row("p1", "relay_p", 20, 2.0, True),
    ]

    summary = paired_bootstrap_summary(
        rows,
        methods=("native_ar", "relay_p"),
        reference_method="native_ar",
        samples=200,
        seed=7,
    )

    relay = summary["methods"]["relay_p"]
    assert relay["throughput_tokens_per_second"] == 10.0
    assert relay["speedup_vs_native_ar"] == 2.0
    assert relay["accuracy"] == 1.0
    assert relay["accuracy_delta_vs_native_ar"] == 0.5
    assert relay["end_to_end_tokens_per_second"] == 30 / 4
    assert relay["mean_input_tokens"] == 7
    assert relay["mean_draft_calls"] == 2
    assert relay["acceptance_survival_by_position"] == {
        "1": 1.0,
        "2": 1.0,
        "3": 0.5,
        "4": 0.5,
    }
    assert relay["ttft_p50_seconds"] == pytest.approx(0.15)
    assert relay["ttft_p95_seconds"] == pytest.approx(0.195)
    assert relay["request_latency_p50_seconds"] == pytest.approx(2.0)
    assert relay["request_latency_p95_seconds"] == pytest.approx(2.45)
    assert relay["allocated_memory_before_bytes"] == 80
    assert relay["peak_allocated_memory_bytes"] == 95
    assert relay["reserved_memory_before_bytes"] == 110
    assert relay["peak_reserved_memory_bytes"] == 130
    assert relay["speedup_vs_native_ar_ci95"][0] == 2.0
    assert relay["speedup_vs_native_ar_ci95"][1] == 2.0


def test_paired_bootstrap_summary_rejects_incomplete_pairs() -> None:
    rows = [
        _row("p0", "native_ar", 10, 2.0, True),
        _row("p0", "relay_p", 10, 1.0, True),
        _row("p1", "native_ar", 10, 2.0, True),
    ]

    try:
        paired_bootstrap_summary(
            rows,
            methods=("native_ar", "relay_p"),
            samples=10,
        )
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("incomplete pairs should fail")


def test_paired_bootstrap_summary_supports_source_only_reference() -> None:
    rows = [
        _row("p0", "source_reuse_eagle3", 10, 2.0, True),
        _row("p0", "relay_eagle3", 10, 1.6, True),
        _row("p1", "source_reuse_eagle3", 20, 4.0, False),
        _row("p1", "relay_eagle3", 20, 3.2, False),
    ]

    summary = paired_bootstrap_summary(
        rows,
        methods=("source_reuse_eagle3", "relay_eagle3"),
        reference_method="source_reuse_eagle3",
        samples=20,
        seed=1,
    )

    assert summary["reference_method"] == "source_reuse_eagle3"
    relay = summary["methods"]["relay_eagle3"]
    assert relay["speedup_vs_reference"] == pytest.approx(1.25)
    assert relay["accuracy_delta_vs_reference"] == 0.0
    assert relay["exact_sequence_match_rate_vs_reference"] == 0.0
    assert relay["speedup_vs_source_reuse"] == pytest.approx(1.25)
    assert "speedup_vs_native_ar" not in relay


def test_paired_bootstrap_summary_omits_placeholder_ttft() -> None:
    rows = [
        _row("p0", "source", 10, 2.0, True),
        _row("p0", "relay", 10, 1.0, True),
    ]
    for row in rows:
        row["time_to_first_token_seconds"] = 0.0

    summary = paired_bootstrap_summary(
        rows,
        methods=("source", "relay"),
        reference_method="source",
        samples=20,
        seed=1,
    )

    assert "ttft_p50_seconds" not in summary["methods"]["source"]
    assert "ttft_p95_seconds" not in summary["methods"]["relay"]


def test_paired_bootstrap_clusters_mtbench_turns_by_conversation() -> None:
    rows = []
    for problem_id in ("mtbench/3/turn0", "mtbench/3/turn1"):
        source = _row(problem_id, "source", 10, 2.0, True)
        relay = _row(problem_id, "relay", 10, 1.0, True)
        source.update({"benchmark": "mtbench", "turn_index": int(problem_id[-1])})
        relay.update({"benchmark": "mtbench", "turn_index": int(problem_id[-1])})
        rows.extend((source, relay))

    summary = paired_bootstrap_summary(
        rows,
        methods=("source", "relay"),
        reference_method="source",
        samples=20,
        seed=1,
    )

    assert summary["paired_requests"] == 2
    assert summary["bootstrap_clusters"] == 1
    assert summary["bootstrap_unit"] == "paired request; MT-Bench conversation"
    assert summary["methods"]["relay"]["speedup_vs_reference_ci95"] == [2.0, 2.0]


def test_paired_binary_delta_summary_retains_task_pairing() -> None:
    result = paired_binary_delta_summary(
        {"a": True, "b": False, "c": False, "d": True},
        {"a": True, "b": True, "c": False, "d": False},
        samples=200,
        seed=7,
    )

    assert result["paired_tasks"] == 4
    assert result["reference_accuracy"] == 0.5
    assert result["candidate_accuracy"] == 0.5
    assert result["accuracy_delta"] == 0.0
    assert result["candidate_only_passes"] == 1
    assert result["reference_only_passes"] == 1
    assert result["accuracy_delta_ci95"][0] <= 0.0
    assert result["accuracy_delta_ci95"][1] >= 0.0


def test_paired_mismatch_audit_keeps_quality_and_divergence_context() -> None:
    rows = [
        {
            **_row("p0", "source", 4, 1.0, True),
            "output_hash": "same",
            "completion": "answer 42",
        },
        {
            **_row("p0", "relay", 4, 0.8, True),
            "output_hash": "same",
            "completion": "answer 42",
        },
        {
            **_row("p1", "source", 5, 1.1, False),
            "output_hash": "source-hash",
            "completion": "boxed{7}",
        },
        {
            **_row("p1", "relay", 6, 0.9, False),
            "output_hash": "relay-hash",
            "completion": "boxed{7} units",
        },
    ]

    audit = paired_mismatch_audit(
        rows, reference_method="source", candidate_method="relay", context_chars=8
    )

    assert audit["paired_requests"] == 2
    assert audit["mismatch_count"] == 1
    mismatch = audit["mismatches"][0]
    assert mismatch["problem_id"] == "p1"
    assert mismatch["common_prefix_chars"] == len("boxed{7}")
    assert mismatch["reference_correct"] is False
    assert mismatch["candidate_correct"] is False
    assert mismatch["reference_tail"] == "boxed{7}"
    assert mismatch["candidate_tail"] == "boxed{7} units"
