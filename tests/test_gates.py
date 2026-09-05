from __future__ import annotations

import pytest

from relayspec.gates import evaluate_development_gate, select_profile_provider


def test_profile_provider_selection_uses_only_no_slowdown_boundaries() -> None:
    assert select_profile_provider(predicted_speedup=1.1) == "relay"
    assert select_profile_provider(predicted_speedup=0.99) == "source"
    assert (
        select_profile_provider(predicted_speedup=1.1, speed_ci_lower=0.99) == "source"
    )
    assert (
        select_profile_provider(predicted_speedup=1.1, speed_ci_lower=1.01) == "relay"
    )


def _analysis() -> dict[str, object]:
    return {
        "paired_requests": 32,
        "exact_sequence_match_rate": 1.0,
        "observed_speedup": 1.2,
        "paired_bootstrap_95_percent": {
            "point_estimate": 1.2,
            "lower": 1.1,
            "upper": 1.3,
        },
        "amdahl_estimate": {
            "acceptance_ratio": 0.9,
            "break_even_acceptance_ratio": 0.8,
            "predicted_speedup": 1.19,
        },
    }


def test_development_gate_accepts_complete_exact_positive_probe() -> None:
    result = evaluate_development_gate(_analysis(), expected_pairs=32)

    assert result["passed"] is True
    assert result["failures"] == []
    assert result["selected_provider"] == "relay"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("paired_requests", 31, "paired request count"),
        ("exact_sequence_match_rate", 31 / 32, "exact sequence agreement"),
        ("observed_speedup", 0.99, "observed speedup"),
    ],
)
def test_development_gate_rejects_failed_primary_condition(
    field: str, value: float, reason: str
) -> None:
    analysis = _analysis()
    analysis[field] = value

    result = evaluate_development_gate(analysis, expected_pairs=32)

    assert result["passed"] is False
    assert result["selected_provider"] == "source"
    assert any(reason in failure for failure in result["failures"])


def test_development_gate_requires_speed_interval_above_break_even() -> None:
    analysis = _analysis()
    interval = analysis["paired_bootstrap_95_percent"]
    assert isinstance(interval, dict)
    interval["lower"] = 0.99

    result = evaluate_development_gate(analysis, expected_pairs=32)

    assert result["passed"] is False
    assert any("lower confidence bound" in failure for failure in result["failures"])


def test_development_gate_requires_acceptance_above_amdahl_break_even() -> None:
    analysis = _analysis()
    amdahl = analysis["amdahl_estimate"]
    assert isinstance(amdahl, dict)
    amdahl["acceptance_ratio"] = 0.79

    result = evaluate_development_gate(analysis, expected_pairs=32)

    assert result["passed"] is False
    assert any("acceptance retention" in failure for failure in result["failures"])
