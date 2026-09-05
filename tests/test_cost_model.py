from __future__ import annotations

import pytest

from relayspec.cost_model import AmdahlInputs, estimate_relay_speedup


def test_cost_model_matches_cycle_acceptance_equation() -> None:
    estimate = estimate_relay_speedup(
        AmdahlInputs(
            cycle_fraction=0.60,
            source_fraction=0.35,
            other_fraction=0.05,
            relay_fraction=0.01,
            source_committed_per_cycle=7.0,
            relay_committed_per_cycle=6.5,
        )
    )

    expected_ratio = (7.0 / 6.5) * 0.60 + 0.05 + 0.01
    assert estimate.latency_ratio == pytest.approx(expected_ratio)
    assert estimate.predicted_speedup == pytest.approx(1 / expected_ratio)
    assert estimate.acceptance_ratio == pytest.approx(6.5 / 7.0)
    assert estimate.break_even_acceptance_ratio == pytest.approx(0.60 / 0.94)
    assert estimate.ideal_equal_acceptance_speedup == pytest.approx(1 / 0.66)


def test_cost_model_rejects_unreconciled_source_profile() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        estimate_relay_speedup(
            AmdahlInputs(
                cycle_fraction=0.5,
                source_fraction=0.2,
                other_fraction=0.2,
                relay_fraction=0.01,
                source_committed_per_cycle=4,
                relay_committed_per_cycle=4,
            )
        )


def test_cost_model_marks_candidate_below_break_even() -> None:
    estimate = estimate_relay_speedup(
        AmdahlInputs(
            cycle_fraction=0.70,
            source_fraction=0.25,
            other_fraction=0.05,
            relay_fraction=0.01,
            source_committed_per_cycle=8,
            relay_committed_per_cycle=5,
        )
    )

    assert not estimate.predicted_faster
    assert estimate.predicted_speedup < 1
