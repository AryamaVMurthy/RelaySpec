from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load_builder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_breadth_matrix.py"
    spec = importlib.util.spec_from_file_location("breadth_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def method(tokens_per_second: float, exact: float = 1.0) -> dict:
    return {
        "end_to_end_tokens_per_second": tokens_per_second,
        "end_to_end_speedup_vs_reference": 1.0,
        "end_to_end_speedup_vs_reference_ci95": [1.0, 1.0],
        "exact_sequence_match_rate_vs_reference": exact,
        "accuracy": 0.75,
        "cap_hit_rate": 0.0,
        "request_latency_p50_seconds": 1.0,
        "request_latency_p95_seconds": 2.0,
        "mean_acceptance_length": 4.0,
    }


def test_build_cell_summary_uses_reference_and_official_code_quality() -> None:
    builder = load_builder().build_cell_summary
    source = method(100.0)
    relay = method(125.0, exact=0.99)
    relay.update(
        {
            "end_to_end_speedup_vs_reference": 1.25,
            "end_to_end_speedup_vs_reference_ci95": [1.2, 1.3],
            "mean_acceptance_length": 3.5,
        }
    )
    paper = {
        "reference_method": "source",
        "by_benchmark": {
            "humaneval": {
                "paired_requests": 164,
                "bootstrap_clusters": 164,
                "methods": {"source": source, "relay": relay},
            }
        },
    }
    evalplus = {
        "benchmarks": {
            "humaneval": {
                "source": {"base_pass_at_1": 0.8, "plus_pass_at_1": 0.7},
                "relay": {"base_pass_at_1": 0.79, "plus_pass_at_1": 0.69},
            }
        },
        "paired_quality": {"humaneval": {"relay": {"plus": {"accuracy_delta": -0.01}}}},
    }

    result = builder(family="eagle3", target="8b", paper=paper, evalplus=evalplus)

    row = result["benchmarks"]["humaneval"]
    assert row["candidate_method"] == "relay"
    assert row["speedup"] == 1.25
    assert row["speedup_ci95"] == [1.2, 1.3]
    assert row["exact_match_rate"] == 0.99
    assert row["acceptance_retention"] == 3.5 / 4.0
    assert row["official_quality"]["source"]["plus_pass_at_1"] == 0.7
    assert row["paired_quality"]["plus"]["accuracy_delta"] == -0.01
    assert result["summary"]["geometric_mean_speedup"] == 1.25
    assert result["summary"]["positive_speed_ci_cells"] == 1


def test_build_cell_summary_requires_expected_breadth_counts() -> None:
    builder = load_builder().build_cell_summary
    paper = {
        "reference_method": "source",
        "by_benchmark": {
            "gsm8k": {
                "paired_requests": 127,
                "bootstrap_clusters": 127,
                "methods": {"source": method(100.0), "relay": method(120.0)},
            }
        },
    }

    try:
        builder(family="dflash", target="8b", paper=paper, evalplus=None)
    except ValueError as error:
        assert "gsm8k" in str(error) and "128" in str(error)
    else:
        raise AssertionError("incomplete breadth benchmark should fail")


def test_build_cell_summary_accepts_complete_task_override() -> None:
    builder = load_builder().build_cell_summary
    incomplete = {
        "reference_method": "source",
        "by_benchmark": {
            "mtbench": {
                "paired_requests": 80,
                "bootstrap_clusters": 80,
                "methods": {"source": method(100.0), "relay": method(120.0)},
            }
        },
    }
    complete = {
        "paired_requests": 160,
        "bootstrap_clusters": 80,
        "methods": {"source": method(100.0), "relay": method(120.0)},
    }

    result = builder(
        family="eagle3",
        target="8b",
        paper=incomplete,
        evalplus=None,
        benchmark_overrides={"mtbench": complete},
    )

    assert result["benchmarks"]["mtbench"]["requests"] == 160
    assert result["benchmarks"]["mtbench"]["bootstrap_clusters"] == 80


def test_amdahl_diagnostics_uses_reference_normalized_components() -> None:
    diagnostics = load_builder().amdahl_diagnostics
    source = method(100.0)
    source.update(
        {
            "output_tokens": 1000,
            "mean_acceptance_length": 99.0,
            "acceptance_survival_by_position": {
                "1": 1.0,
                "2": 1.0,
                "3": 1.0,
                "4": 1.0,
                "5": 1.0,
            },
            "profile_region_totals_ms": {
                "draft": 2000.0,
                "verification_full_target": 3000.0,
                "prefill_source_trunk": 100.0,
                "verification_source_trunk": 2900.0,
                "unattributed_runtime": 2000.0,
            },
        }
    )
    relay = method(120.0)
    relay.update(
        {
            "mean_acceptance_length": 1.0,
            "acceptance_survival_by_position": {
                "1": 1.0,
                "2": 1.0,
                "3": 1.0,
                "4": 1.0,
                "5": 0.5,
            },
            "profile_region_totals_ms": {
                "prefill_relay": 10.0,
                "relay": 90.0,
            },
        }
    )

    result = diagnostics(source, relay)

    assert result["source_fraction"] == 0.3
    assert result["cycle_fraction"] == 0.5
    assert result["other_fraction"] == pytest.approx(0.2)
    assert result["relay_fraction"] == 0.01
    assert result["acceptance_retention"] == 0.9
    assert result["predicted_speedup"] == pytest.approx(
        1 / ((1 / 0.9) * 0.5 + 0.2 + 0.01)
    )


def test_cell_summary_applies_coefficient_free_profile_provider_policy() -> None:
    builder = load_builder().build_cell_summary
    source = method(100.0)
    source.update(
        {
            "output_tokens": 1000,
            "acceptance_survival_by_position": {"1": 1.0, "2": 1.0},
            "profile_region_totals_ms": {
                "draft": 2000.0,
                "verification_full_target": 4000.0,
                "prefill_source_trunk": 100.0,
                "verification_source_trunk": 2900.0,
                "unattributed_runtime": 1000.0,
            },
        }
    )
    relay = method(90.0)
    relay.update(
        {
            "end_to_end_speedup_vs_reference": 0.9,
            "end_to_end_speedup_vs_reference_ci95": [0.88, 0.92],
            "acceptance_survival_by_position": {"1": 1.0, "2": 0.2},
            "profile_region_totals_ms": {"prefill_relay": 10.0, "relay": 90.0},
        }
    )
    paper = {
        "reference_method": "source",
        "by_benchmark": {
            "gsm8k": {
                "paired_requests": 128,
                "bootstrap_clusters": 128,
                "methods": {"source": source, "relay": relay},
            }
        },
    }

    result = builder(family="eagle3", target="14b", paper=paper, evalplus=None)

    row = result["benchmarks"]["gsm8k"]
    assert row["profile_policy"]["selected_provider"] == "source"
    assert row["profile_policy"]["realized_speedup"] == 1.0
    assert row["profile_policy"]["direction_match"] is True
    assert result["summary"]["profile_policy_geometric_mean_speedup"] == 1.0
    assert result["summary"]["profile_policy_direction_matches"] == 1


def test_profile_policy_keeps_source_when_speed_interval_crosses_one() -> None:
    builder = load_builder().build_cell_summary
    source = method(100.0)
    source.update(
        {
            "output_tokens": 1000,
            "acceptance_survival_by_position": {"1": 1.0, "2": 1.0},
            "profile_region_totals_ms": {
                "draft": 2000.0,
                "verification_full_target": 4000.0,
                "prefill_source_trunk": 100.0,
                "verification_source_trunk": 2900.0,
                "unattributed_runtime": 1000.0,
            },
        }
    )
    relay = method(102.0)
    relay.update(
        {
            "end_to_end_speedup_vs_reference": 1.02,
            "end_to_end_speedup_vs_reference_ci95": [0.99, 1.05],
            "acceptance_survival_by_position": {"1": 1.0, "2": 0.9},
            "profile_region_totals_ms": {"prefill_relay": 10.0, "relay": 90.0},
        }
    )
    paper = {
        "reference_method": "source",
        "by_benchmark": {
            "gsm8k": {
                "paired_requests": 128,
                "bootstrap_clusters": 128,
                "methods": {"source": source, "relay": relay},
            }
        },
    }

    result = builder(family="eagle3", target="14b", paper=paper, evalplus=None)

    policy = result["benchmarks"]["gsm8k"]["profile_policy"]
    assert policy["selected_provider"] == "source"
    assert policy["realized_speedup"] == 1.0
    assert policy["point_direction_match"] is True
    assert policy["speed_evidence_passes"] is False


def test_aggregate_summary_covers_every_benchmark_cell() -> None:
    aggregate = load_builder().aggregate_summary
    cells = [
        {
            "benchmarks": {
                "gsm8k": {
                    "speedup": 1.2,
                    "speedup_ci95": [1.1, 1.3],
                    "amdahl": {"predicted_speedup": 1.19},
                    "profile_policy": {
                        "realized_speedup": 1.2,
                        "direction_match": True,
                    },
                },
                "humaneval": {
                    "speedup": 0.8,
                    "speedup_ci95": [0.7, 0.9],
                    "amdahl": {"predicted_speedup": 0.81},
                    "profile_policy": {
                        "realized_speedup": 1.0,
                        "direction_match": True,
                    },
                },
            }
        }
    ]

    result = aggregate(cells)

    assert result["benchmark_cells"] == 2
    assert result["positive_speed_ci_cells"] == 1
    assert result["amdahl_direction_matches"] == 2
    assert result["raw_geometric_mean_speedup"] == pytest.approx((1.2 * 0.8) ** 0.5)
    assert result["profile_policy_geometric_mean_speedup"] == pytest.approx(1.2**0.5)
