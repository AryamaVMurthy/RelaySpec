import pytest
import torch

from relayspec.sampling_rng import initialize_sampling_rng


def test_sampling_is_independent_of_prior_rng_consumption():
    seed = initialize_sampling_rng(0.6, 1729, "gsm8k/example", 0, 0)
    first = torch.multinomial(torch.tensor([0.2, 0.3, 0.5]), 100, replacement=True)
    torch.rand(1000)
    assert initialize_sampling_rng(0.6, 1729, "gsm8k/example", 0, 0) == seed
    torch.testing.assert_close(
        first, torch.multinomial(torch.tensor([0.2, 0.3, 0.5]), 100, replacement=True)
    )
    alternatives = [
        initialize_sampling_rng(0.6, base, pid, rep, turn)
        for base, pid, rep, turn in [
            (1730, "gsm8k/example", 0, 0),
            (1729, "gsm8k/other", 0, 0),
            (1729, "gsm8k/example", 1, 0),
            (1729, "gsm8k/example", 0, 1),
        ]
    ]
    assert len(set([seed, *alternatives])) == 5


def test_greedy_does_not_reset_rng_and_sampling_requires_declared_seed():
    before = torch.random.get_rng_state().clone()
    assert initialize_sampling_rng(0.0, None, "example", 0, 0) is None
    torch.testing.assert_close(before, torch.random.get_rng_state())
    for bad in [None, True, -1, "1729"]:
        with pytest.raises(ValueError):
            initialize_sampling_rng(0.6, bad, "example", 0, 0)
