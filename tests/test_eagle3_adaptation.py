import pytest
import torch

from relayspec.eagle3_adaptation import (
    eagle_teacher_forced_logits,
    shifted_eagle_inputs,
)


class CausalDraft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)

    def forward(
        self, *, hidden_states, input_ids, position_ids, past_key_values, use_cache
    ):
        assert past_key_values is None and not use_cache
        assert torch.equal(position_ids, torch.arange(input_ids.shape[1]).unsqueeze(0))
        return (hidden_states + input_ids.unsqueeze(-1)).cumsum(dim=1) * self.scale

    def compute_logits(self, hidden):
        return torch.cat([hidden, -hidden], dim=-1)


def test_eagle_shift_matches_target_label_positions_and_keeps_mapper_gradients():
    # Input at source index t+1 combines with feature t to predict target index t+2.
    ids = torch.arange(20).unsqueeze(0)
    features = torch.arange(20.0).reshape(1, 20, 1)
    x, tokens, positions = shifted_eagle_inputs(features, ids)
    assert torch.equal(x, features[:, :-2])
    assert torch.equal(tokens, ids[:, 1:-1])
    prediction_token_positions = tokens[:, -15:] + 1
    target_tail_logit_positions = ids[:, -16:-1]
    assert torch.equal(prediction_token_positions, target_tail_logit_positions + 1)
    mapper = torch.nn.Linear(1, 1, bias=False)
    draft = CausalDraft()
    logits = eagle_teacher_forced_logits(draft, mapper, x, tokens, positions)
    assert logits.shape == (1, 15, 2)
    logits[..., 0].sum().backward()
    assert mapper.weight.grad is not None and mapper.weight.grad.abs().sum() > 0
    assert draft.scale.grad is None
    changed = tokens.clone()
    changed[:, -1] = 999
    other = eagle_teacher_forced_logits(draft, mapper, x, changed, positions)
    assert torch.equal(logits[:, :-1], other[:, :-1])
    assert not torch.equal(logits[:, -1], other[:, -1])


def test_eagle_rejects_missing_or_misaligned_supervision():
    with pytest.raises(ValueError, match="aligned"):
        shifted_eagle_inputs(torch.ones(1, 20, 2), torch.ones(1, 19, dtype=torch.long))
    with pytest.raises(ValueError, match="label block"):
        shifted_eagle_inputs(torch.ones(1, 16, 2), torch.ones(1, 16, dtype=torch.long))
    with pytest.raises(ValueError, match="must align"):
        eagle_teacher_forced_logits(
            CausalDraft(),
            torch.nn.Identity(),
            torch.ones(1, 14, 1),
            torch.ones(1, 14, dtype=torch.long),
            torch.arange(14).unsqueeze(0),
        )
