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


def test_detailed_observer_distinguishes_physical_cache_from_valid_prefix():
    model = SimpleNamespace(
        greedy_sample=True, _relayspec_trace=[], _relayspec_detailed_trace=True, NG=2
    )
    ids = torch.tensor([[7, 6, 8, 1, 3, 2, 4]])
    mask = torch.tensor([[1, 0, 1, 1, 1, 1, 1]])
    positions = torch.tensor([[0, 0, 1, 2, 3, 4, 5]])
    logits = torch.nn.functional.one_hot(torch.tensor([[3, 2, 4]]), 9).float()
    capture_sd_square_step(
        model,
        ids,
        3,
        torch.tensor([2]),
        torch.tensor([[4]]),
        logits,
        torch.tensor([False]),
        mask,
        positions,
    )
    row = model._relayspec_trace[0]
    assert row["physical_cache_prefix"] == 3
    assert row["valid_cached_prefix_ids"] == [7, 8]
    assert row["query_ids"] == [1, 3, 2]
    assert row["query_positions"] == [2, 3, 4]
    assert row["output_start"] == 0
