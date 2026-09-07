"""Guard the released DFlash distinction between proposal and target sampling."""

from types import SimpleNamespace

import pytest
import torch

from relayspec import generation


@pytest.mark.parametrize("temperature", [0.0, 0.6, 1.0])
def test_relay_proposes_greedily_but_preserves_target_temperature(
    monkeypatch, temperature
):
    class Cache:
        def __init__(self, **kwargs):
            pass

        def get_seq_length(self):
            return 0

        def crop(self, length):
            pass

    class Target(torch.nn.Module):
        config = SimpleNamespace()

        def forward(self, ids, **kwargs):
            states = torch.nn.functional.one_hot(ids, num_classes=3).float()
            logits = torch.zeros_like(states)
            logits[..., 0] = 10
            if kwargs.get("logits_to_keep"):
                logits = logits[:, -kwargs["logits_to_keep"] :, :]
            return SimpleNamespace(logits=logits, hidden_states=(states, states))

    monkeypatch.setattr("transformers.DynamicCache", Cache)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        generation,
        "_conditioned_dflash_forward",
        lambda draft, **kwargs: kwargs["noise_embedding"],
    )
    observed = []

    def sample(logits, temperature=0.0):
        observed.append(temperature)
        return logits.argmax(dim=-1)

    monkeypatch.setattr(generation, "greedy_sample", sample)
    draft = torch.nn.Module()
    draft.block_size = 2
    draft.mask_token_id = 1
    draft.hidden_norm = torch.nn.Identity()
    output = generation.relay_dflash_generate(
        draft,
        relay=torch.nn.Identity(),
        relay_target_layer_ids=(0,),
        native_target=Target(),
        source_embedding=torch.nn.Embedding(3, 3),
        source_lm_head=torch.nn.Identity(),
        input_ids=torch.tensor([[0, 0]]),
        max_new_tokens=4,
        stop_token_ids=None,
        temperature=temperature,
    )
    assert output.shape == (1, 6)
    assert observed[0] == temperature  # Target prefill.
    assert len(observed) >= 3 and len(observed[1:]) % 2 == 0
    assert all(t == 0.0 for t in observed[1::2])  # Proposal calls.
    assert all(t == temperature for t in observed[2::2])  # Target verification.
