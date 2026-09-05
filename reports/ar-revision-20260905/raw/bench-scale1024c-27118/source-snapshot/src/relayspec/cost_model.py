from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AmdahlInputs:
    cycle_fraction: float
    source_fraction: float
    other_fraction: float
    relay_fraction: float
    source_committed_per_cycle: float
    relay_committed_per_cycle: float


@dataclass(frozen=True)
class AmdahlEstimate:
    latency_ratio: float
    predicted_speedup: float
    acceptance_ratio: float
    break_even_acceptance_ratio: float
    ideal_equal_acceptance_speedup: float
    predicted_faster: bool


def estimate_relay_speedup(inputs: AmdahlInputs) -> AmdahlEstimate:
    source_components = (
        inputs.cycle_fraction + inputs.source_fraction + inputs.other_fraction
    )
    if abs(source_components - 1.0) > 1e-6:
        raise ValueError("source profile fractions must sum to one")
    for name in ("cycle_fraction", "source_fraction", "other_fraction"):
        if getattr(inputs, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if inputs.relay_fraction < 0:
        raise ValueError("relay_fraction must be non-negative")
    if inputs.source_committed_per_cycle <= 0:
        raise ValueError("source_committed_per_cycle must be positive")
    if inputs.relay_committed_per_cycle <= 0:
        raise ValueError("relay_committed_per_cycle must be positive")

    acceptance_ratio = (
        inputs.relay_committed_per_cycle / inputs.source_committed_per_cycle
    )
    latency_ratio = (
        inputs.source_committed_per_cycle
        / inputs.relay_committed_per_cycle
        * inputs.cycle_fraction
        + inputs.other_fraction
        + inputs.relay_fraction
    )
    break_even_denominator = 1.0 - inputs.other_fraction - inputs.relay_fraction
    if break_even_denominator <= 0:
        raise ValueError("other and relay work leave no break-even budget")
    break_even = inputs.cycle_fraction / break_even_denominator
    equal_acceptance_denominator = 1.0 - inputs.source_fraction + inputs.relay_fraction
    if equal_acceptance_denominator <= 0:
        raise ValueError("invalid equal-acceptance latency denominator")
    predicted_speedup = 1.0 / latency_ratio
    return AmdahlEstimate(
        latency_ratio=latency_ratio,
        predicted_speedup=predicted_speedup,
        acceptance_ratio=acceptance_ratio,
        break_even_acceptance_ratio=break_even,
        ideal_equal_acceptance_speedup=1.0 / equal_acceptance_denominator,
        predicted_faster=predicted_speedup > 1.0,
    )
