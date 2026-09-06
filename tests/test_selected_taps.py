import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from relayspec.relay import extract_hidden_taps
from relayspec.selected_taps import forward_selected_taps


def tiny_target():
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        head_dim=4, attention_dropout=0.0,
    )).eval()


@torch.inference_mode()
def test_selected_taps_match_full_states_including_final_normalization():
    model = tiny_target()
    ids = torch.tensor([[1, 2, 3, 4]])
    full = model(ids, output_hidden_states=True, use_cache=False)
    previous_layer_hooks = dict(model.model.layers[0]._forward_hooks)
    previous_norm_hooks = dict(model.model.norm._forward_hooks)
    output, taps = forward_selected_taps(model, ids, [0, 2], use_cache=False)
    assert output.hidden_states is None
    assert torch.equal(output.logits, full.logits)
    assert torch.equal(torch.cat(taps, dim=-1), extract_hidden_taps(full.hidden_states, [0, 2]))
    assert dict(model.model.layers[0]._forward_hooks) == previous_layer_hooks
    assert dict(model.model.norm._forward_hooks) == previous_norm_hooks


def test_hooks_are_removed_after_target_failure(monkeypatch):
    model = tiny_target()
    def fail(*_args, **_kwargs):
        raise RuntimeError("injected failure")
    monkeypatch.setattr(model.model.layers[1], "forward", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        forward_selected_taps(model, torch.tensor([[1, 2]]), [0, 2])
    assert not model.model.layers[0]._forward_hooks
    assert not model.model.norm._forward_hooks


def test_invalid_layer_selection_fails_before_registering_hooks():
    model = tiny_target()
    with pytest.raises(ValueError, match="outside"):
        forward_selected_taps(model, torch.tensor([[1]]), [3])
    assert not any(layer._forward_hooks for layer in model.model.layers)
