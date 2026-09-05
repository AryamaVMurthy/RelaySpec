from __future__ import annotations

from typing import Any


def evaluate_development_gate(
    analysis: dict[str, Any], *, expected_pairs: int
) -> dict[str, Any]:
    """Evaluate the predeclared relay development gate.

    Every cutoff has a direct interpretation: complete coverage, no observed
    paired output change, and strict improvement over the unit speed and
    acceptance break-even boundaries.  No test-set quality enters this gate.
    """
    failures: list[str] = []
    paired_requests = int(analysis["paired_requests"])
    exact_rate = float(analysis["exact_sequence_match_rate"])
    observed_speedup = float(analysis["observed_speedup"])
    interval = analysis["paired_bootstrap_95_percent"]
    amdahl = analysis["amdahl_estimate"]
    lower_speedup = float(interval["lower"])
    acceptance_ratio = float(amdahl["acceptance_ratio"])
    break_even_acceptance = float(amdahl["break_even_acceptance_ratio"])
    predicted_speedup = float(amdahl["predicted_speedup"])

    if paired_requests != expected_pairs:
        failures.append(
            f"paired request count {paired_requests} != expected {expected_pairs}"
        )
    if exact_rate != 1.0:
        failures.append(f"exact sequence agreement {exact_rate:.6f} != 1.0")
    if observed_speedup <= 1.0:
        failures.append(f"observed speedup {observed_speedup:.6f} <= 1.0")
    if lower_speedup <= 1.0:
        failures.append(
            f"paired speed lower confidence bound {lower_speedup:.6f} <= 1.0"
        )
    if acceptance_ratio <= break_even_acceptance:
        failures.append(
            "acceptance retention "
            f"{acceptance_ratio:.6f} <= Amdahl break-even "
            f"{break_even_acceptance:.6f}"
        )
    if predicted_speedup <= 1.0:
        failures.append(f"Amdahl-predicted speedup {predicted_speedup:.6f} <= 1.0")

    return {
        "passed": not failures,
        "failures": failures,
        "criteria": {
            "expected_pairs": expected_pairs,
            "required_exact_sequence_match_rate": 1.0,
            "required_observed_speedup_strictly_above": 1.0,
            "required_speed_ci_lower_strictly_above": 1.0,
            "required_acceptance_ratio_strictly_above_break_even": (
                break_even_acceptance
            ),
        },
        "observed": {
            "paired_requests": paired_requests,
            "exact_sequence_match_rate": exact_rate,
            "observed_speedup": observed_speedup,
            "speed_ci_lower": lower_speedup,
            "acceptance_ratio": acceptance_ratio,
            "break_even_acceptance_ratio": break_even_acceptance,
            "amdahl_predicted_speedup": predicted_speedup,
        },
    }
