from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_benchmarking():
    path = Path(__file__).resolve().parents[1] / "src" / "relayspec" / "benchmarking.py"
    spec = importlib.util.spec_from_file_location("relayspec_benchmarking", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shard_records_covers_each_record_once() -> None:
    shard_records = load_benchmarking().shard_records

    records = [{"problem_id": f"p{index}"} for index in range(11)]
    shards = [shard_records(records, world_size=4, rank=rank) for rank in range(4)]

    observed = [row["problem_id"] for shard in shards for row in shard]
    assert sorted(observed) == sorted(row["problem_id"] for row in records)
    assert [len(shard) for shard in shards] == [3, 3, 3, 2]


def test_shard_records_rejects_invalid_rank() -> None:
    shard_records = load_benchmarking().shard_records

    with pytest.raises(ValueError, match="rank"):
        shard_records([{"problem_id": "p0"}], world_size=4, rank=4)


def test_rotate_methods_changes_first_method_without_changing_members() -> None:
    rotate_methods = load_benchmarking().rotate_methods

    methods = ("native_ar", "naive_source_reuse", "relay_p")
    assert rotate_methods(methods, 0) == methods
    assert rotate_methods(methods, 1) == (
        "naive_source_reuse",
        "relay_p",
        "native_ar",
    )
    assert rotate_methods(methods, 2) == (
        "relay_p",
        "native_ar",
        "naive_source_reuse",
    )


def test_format_benchmark_prompt_only_adds_math_instruction_to_math_tasks() -> None:
    format_benchmark_prompt = load_benchmarking().format_benchmark_prompt

    math = format_benchmark_prompt({"benchmark": "math500", "prompt": "2+2?"})
    code = format_benchmark_prompt({"benchmark": "humaneval", "prompt": "def f():"})

    assert "\\boxed{}" in math
    assert code == "def f():"


def test_benchmark_turns_preserves_mtbench_turn_order() -> None:
    benchmark_turns = load_benchmarking().benchmark_turns

    assert benchmark_turns({"benchmark": "mtbench", "turns": ["first", "second"]}) == (
        "first",
        "second",
    )


def test_generation_call_counts_uses_explicit_instrumentation() -> None:
    generation_call_counts = load_benchmarking().generation_call_counts
    stats = SimpleNamespace(
        acceptance_lengths=[3, 4],
        target_calls=7,
        draft_calls=5,
    )

    assert generation_call_counts(stats) == (7, 5)


def test_generation_call_counts_supports_official_dflash_stats() -> None:
    generation_call_counts = load_benchmarking().generation_call_counts
    stats = SimpleNamespace(acceptance_lengths=[4, 5, 2])

    assert generation_call_counts(stats) == (3, 3)


def test_optional_generation_stats_preserves_provider_and_eagle_counts() -> None:
    optional_generation_stats = load_benchmarking().optional_generation_stats
    stats = SimpleNamespace(
        source_executed_tokens=37,
        proposal_lengths=[3, 3],
        accepted_draft_lengths=[2, 1],
        verify_count=2,
    )

    assert optional_generation_stats(stats) == {
        "source_executed_tokens": 37,
        "proposal_lengths": [3, 3],
        "accepted_draft_lengths": [2, 1],
        "verify_count": 2,
    }


def test_generation_timings_fall_back_to_measured_request_for_official_api() -> None:
    generation_timings = load_benchmarking().generation_timings

    ttft, decode = generation_timings(
        SimpleNamespace(num_output_tokens=10),
        request_seconds=2.5,
    )

    assert ttft == 0.0
    assert decode == 2.5


def test_aggregate_rows_uses_native_output_as_exactness_reference() -> None:
    aggregate_rows = load_benchmarking().aggregate_rows

    rows = [
        {
            "problem_id": "p0",
            "repetition": 0,
            "method": "native_ar",
            "output_hash": "a",
            "output_tokens": 10,
            "decode_seconds": 2.0,
            "acceptance_length": 1.0,
        },
        {
            "problem_id": "p0",
            "repetition": 0,
            "method": "naive_source_reuse",
            "output_hash": "a",
            "output_tokens": 10,
            "decode_seconds": 1.0,
            "acceptance_length": 5.0,
        },
        {
            "problem_id": "p0",
            "repetition": 0,
            "method": "relay_p",
            "output_hash": "a",
            "output_tokens": 10,
            "decode_seconds": 0.8,
            "acceptance_length": 4.5,
        },
        {
            "problem_id": "p1",
            "repetition": 0,
            "method": "native_ar",
            "output_hash": "b",
            "output_tokens": 20,
            "decode_seconds": 4.0,
            "acceptance_length": 1.0,
        },
        {
            "problem_id": "p1",
            "repetition": 0,
            "method": "naive_source_reuse",
            "output_hash": "b",
            "output_tokens": 20,
            "decode_seconds": 2.0,
            "acceptance_length": 6.0,
        },
        {
            "problem_id": "p1",
            "repetition": 0,
            "method": "relay_p",
            "output_hash": "wrong",
            "output_tokens": 20,
            "decode_seconds": 1.6,
            "acceptance_length": 5.0,
        },
    ]

    summary = aggregate_rows(
        rows,
        methods=("native_ar", "naive_source_reuse", "relay_p"),
    )

    assert summary["methods"]["native_ar"]["decode_tokens_per_second"] == 5.0
    assert summary["methods"]["naive_source_reuse"]["speedup_vs_native_ar"] == 2.0
    assert summary["methods"]["naive_source_reuse"]["exact_sequence_matches"] == 2
    assert summary["methods"]["relay_p"]["exact_sequence_matches"] == 1
    assert summary["methods"]["relay_p"]["exact_sequence_match_rate"] == 0.5
    comparison = summary["pairwise"]["relay_p_vs_naive_source_reuse"]
    assert comparison["throughput_speedup"] == pytest.approx(1.25)
    assert comparison["exact_sequence_matches"] == 1
    assert comparison["exact_sequence_match_rate"] == 0.5


def test_aggregate_rows_rejects_incomplete_pairing() -> None:
    aggregate_rows = load_benchmarking().aggregate_rows

    rows = [
        {
            "problem_id": "p0",
            "repetition": 0,
            "method": "native_ar",
            "output_hash": "a",
            "output_tokens": 10,
            "decode_seconds": 2.0,
            "acceptance_length": 1.0,
        },
        {
            "problem_id": "p0",
            "repetition": 0,
            "method": "relay_p",
            "output_hash": "a",
            "output_tokens": 10,
            "decode_seconds": 1.0,
            "acceptance_length": 4.0,
        },
    ]

    with pytest.raises(ValueError, match="incomplete paired benchmark"):
        aggregate_rows(
            rows,
            methods=("native_ar", "naive_source_reuse", "relay_p"),
        )


def test_paired_bootstrap_speedup_is_exact_for_constant_request_ratio() -> None:
    paired_bootstrap_speedup = load_benchmarking().paired_bootstrap_speedup
    rows = []
    for index, reference_seconds in enumerate((2.0, 4.0, 6.0)):
        rows.extend(
            [
                {
                    "problem_id": f"p{index}",
                    "repetition": 0,
                    "method": "source",
                    "request_seconds": reference_seconds,
                },
                {
                    "problem_id": f"p{index}",
                    "repetition": 0,
                    "method": "relay",
                    "request_seconds": reference_seconds / 2,
                },
            ]
        )

    interval = paired_bootstrap_speedup(
        rows,
        reference_method="source",
        candidate_method="relay",
        replicates=100,
        seed=7,
    )

    assert interval == {"estimate": 2.0, "lower": 2.0, "upper": 2.0}


def test_model_load_plan_allows_relay_only_source_trunk_unload() -> None:
    model_load_plan = load_benchmarking().model_load_plan

    plan = model_load_plan(("relay_f",), unload_source_trunk=True)

    assert plan == {
        "load_source": True,
        "build_source_provider": False,
        "load_native_target_draft": False,
        "unload_source_trunk": True,
    }


def test_model_load_plan_rejects_unloading_source_needed_by_baseline() -> None:
    model_load_plan = load_benchmarking().model_load_plan

    with pytest.raises(ValueError, match="naive_source_reuse"):
        model_load_plan(
            ("naive_source_reuse", "relay_f"),
            unload_source_trunk=True,
        )


def test_model_load_plan_skips_unused_optional_models() -> None:
    model_load_plan = load_benchmarking().model_load_plan

    assert model_load_plan(("native_ar",), unload_source_trunk=False) == {
        "load_source": False,
        "build_source_provider": False,
        "load_native_target_draft": False,
        "unload_source_trunk": False,
    }


def test_model_load_plan_recognizes_optimized_source_reuse_name() -> None:
    model_load_plan = load_benchmarking().model_load_plan

    plan = model_load_plan(("optimized_source_reuse",), unload_source_trunk=False)

    assert plan["load_source"] is True
    assert plan["build_source_provider"] is True


def test_eagle3_load_plan_avoids_incompatible_14b_model_residency() -> None:
    eagle3_load_plan = load_benchmarking().eagle3_load_plan

    assert eagle3_load_plan(("native_ar", "native_target_eagle3")) == {
        "load_source": False,
        "load_source_draft": False,
        "load_target_draft": True,
        "load_relay": False,
    }
    assert eagle3_load_plan(("source_reuse_eagle3", "relay_eagle3")) == {
        "load_source": True,
        "load_source_draft": True,
        "load_target_draft": False,
        "load_relay": True,
    }
