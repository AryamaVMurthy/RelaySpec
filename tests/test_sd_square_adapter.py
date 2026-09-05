from types import SimpleNamespace

import pytest
import torch

from relayspec.sd_square_adapter import capture_sd_square_step, committed_tokens


def test_observer_keeps_eos_before_public_mask_clears_finished_block():
    model = SimpleNamespace(greedy_sample=True, _relayspec_trace=[])
    ids = torch.tensor([[7, 3, 2, 4]])
    logits = torch.nn.functional.one_hot(torch.tensor([[3, 2, 4]]), 8).float()
    original = ids.clone()
    mask = torch.ones_like(ids)
    capture_sd_square_step(
        model,
        ids,
        0,
        torch.tensor([2]),
        torch.tensor([[4]]),
        logits,
        torch.tensor([False]),
        mask,
        torch.arange(4)[None],
    )
    mask[:] = 0  # Public code can clear an entire finished block afterwards.
    counted, raw = committed_tokens(model._relayspec_trace, 64, 2)
    assert counted == [3, 2]
    assert raw == [3, 2, 4]
    assert torch.equal(original, ids)
    assert committed_tokens(model._relayspec_trace, 1, 2)[0] == [3]


def test_observer_rejects_token_not_selected_by_greedy_verifier():
    model = SimpleNamespace(greedy_sample=True, _relayspec_trace=[])
    ids = torch.tensor([[7, 3]])
    with pytest.raises(ValueError, match="differing from its greedy verifier"):
        capture_sd_square_step(
            model,
            ids,
            0,
            torch.tensor([0]),
            torch.tensor([[4]]),
            torch.tensor([[[0.0, 1.0, 0.0, 0.0, 0.0]]]),
            torch.tensor([False]),
            torch.ones_like(ids),
            torch.arange(2)[None],
        )
    assert not model._relayspec_trace


def test_ended_sequence_adds_no_further_tokens():
    model = SimpleNamespace(greedy_sample=True, _relayspec_trace=[])
    ids = torch.tensor([[7, 3]])
    capture_sd_square_step(
        model,
        ids,
        0,
        torch.tensor([0]),
        torch.tensor([[4]]),
        torch.zeros(1, 1, 5),
        torch.tensor([True]),
        torch.ones_like(ids),
        torch.arange(2)[None],
    )
    assert not model._relayspec_trace
