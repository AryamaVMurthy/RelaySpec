from types import SimpleNamespace

import pytest
import torch

from relayspec.sd_square_adapter import (
    capture_sd_square_step,
    collate_sd_square_records,
    committed_tokens,
    finalize_sd_square_trace,
    load_inference_steering,
    steering_digest,
    termination_status,
)


def test_public_slot_guard_distinguishes_short_warmup_from_eos_or_cap():
    trace = [{"tokens": [3], "accepted_draft_tokens": 0} for _ in range(7)]
    result = termination_status(trace, 16, 2, 8)
    assert result["reason"] == "public_physical_slot_guard"
    assert result["counted_tokens"] == 7 and result["public_cycle_limit"] == 7
    assert termination_status(trace[:6], 16, 2, 8)["reason"] == "unexplained_early_stop"
    assert termination_status([{"tokens": [3, 2]}], 16, 2, 8)["reason"] == "eos"
    assert (
        termination_status([{"tokens": list(range(16))}], 16, 100, 8)["reason"] == "cap"
    )


def test_inference_conversion_checks_training_bytes_before_bf16_rounding():
    saved = {"weight": torch.tensor([1.0001, 2.0001], dtype=torch.float32)}
    named = [("weight", torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16)))]
    training_sha = steering_digest(list(saved.items()))
    identity = load_inference_steering(named, saved, training_sha)
    assert identity["training_sha256"] == training_sha
    assert identity["inference_sha256"] == steering_digest(named) != training_sha
    assert torch.equal(named[0][1], saved["weight"].bfloat16())
    # This change vanishes under BF16 rounding but still must fail provenance.
    saved["weight"][0] += 0.0001
    assert torch.equal(named[0][1], saved["weight"].bfloat16())
    with pytest.raises(ValueError, match="training checkpoint fingerprint"):
        load_inference_steering(named, saved, training_sha)


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


def test_collation_preserves_records_and_excludes_padding_from_public_loss_mask():
    records = [torch.tensor([[9, 2, 3]]), torch.tensor([[4, 5]])]
    batch, lengths = collate_sd_square_records(records, pad_token_id=9)
    assert lengths == [3, 2]
    assert batch["targets"].tolist() == [[9, 2, 3], [4, 5, 9]]
    assert batch["loss_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert records[1].tolist() == [[4, 5]]
    with pytest.raises(ValueError, match="capped at192"):
        collate_sd_square_records([torch.ones(1, 193, dtype=torch.long)], 9)


def test_deferred_observer_preserves_decisions_after_source_buffers_change():
    model = SimpleNamespace(
        greedy_sample=True,
        _relayspec_trace=[],
        _relayspec_tensor_trace=[],
        _relayspec_capture_tensors=True,
        NG=2,
    )
    ids = torch.tensor([[7, 3, 2, 4]])
    logits = torch.nn.functional.one_hot(torch.tensor([[3, 2, 4]]), 8).float()
    capture_sd_square_step(
        model,
        ids,
        0,
        torch.tensor([2]),
        torch.tensor([[4]]),
        logits,
        torch.tensor([False]),
        torch.ones_like(ids),
        torch.arange(4)[None],
    )
    ids.zero_()
    logits.zero_()
    trace = finalize_sd_square_trace(model)
    assert committed_tokens(trace, 64, 2) == ([3, 2], [3, 2, 4])
    assert trace[0]["verifier_argmax"] == [3, 2, 4]
