import numpy as np
import pytest
from scipy.stats import multinomial

from relayspec.paired_accuracy import paired_accuracy_interval


def test_no_discordances_do_not_prove_one_point_margin_at_256_requests():
    result = paired_accuracy_interval([1] * 256, [1] * 256)
    boundary = 1 - 0.0125 ** (1 / 256)
    assert result["confidence_interval"] == pytest.approx([-boundary, boundary])
    assert result["confidence_interval"][0] < -0.01
    assert result["accuracy_difference"] == 0


def test_pairing_and_reversal_are_preserved():
    same = paired_accuracy_interval([1, 1, 0, 0], [1, 1, 0, 0])
    crossed = paired_accuracy_interval([1, 1, 0, 0], [0, 0, 1, 1])
    assert same["accuracy_difference"] == crossed["accuracy_difference"] == 0
    assert same["adverse_discordances"] == 0
    assert crossed["adverse_discordances"] == 2
    assert crossed["confidence_interval"][1] > same["confidence_interval"][1]
    a = paired_accuracy_interval([1, 1, 1, 0], [0, 0, 1, 1])
    b = paired_accuracy_interval([0, 0, 1, 1], [1, 1, 1, 0])
    assert b["confidence_interval"] == pytest.approx(
        [-x for x in a["confidence_interval"][::-1]]
    )


def test_enumerated_sampling_coverage_on_small_multinomial_grid():
    n = 8
    possibilities = []
    for plus in range(n + 1):
        for minus in range(n - plus + 1):
            ties = n - plus - minus
            result = paired_accuracy_interval(
                [1] * plus + [0] * minus + [1] * ties,
                [0] * plus + [1] * minus + [1] * ties,
            )
            possibilities.append(([plus, minus, ties], result["confidence_interval"]))
    for plus in np.linspace(0, 1, 6):
        for minus in np.linspace(0, 1 - plus, 6):
            probability = [plus, minus, 1 - plus - minus]
            coverage = sum(
                multinomial.pmf(counts, n, probability)
                for counts, (low, high) in possibilities
                if low - 1e-12 <= plus - minus <= high + 1e-12
            )
            assert coverage >= 0.95 - 1e-12


@pytest.mark.parametrize(
    "candidate,reference", [([], []), ([1], [1, 0]), ([2], [1]), ([None], [0])]
)
def test_invalid_or_missing_pairs_fail(candidate, reference):
    with pytest.raises(ValueError):
        paired_accuracy_interval(candidate, reference)
