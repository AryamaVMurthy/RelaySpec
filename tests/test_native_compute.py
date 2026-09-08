import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from relayspec.native_compute import FullLinear, fit_linear_mlp, fit_pruned_mlp, native_intervention


@pytest.fixture
def native_fixture(monkeypatch):
    module = ModuleType("dflash.model")
    calls = []
    def attention(layer, q, k, v, mask, **kwargs):
        calls.append(k.clone())
        return k, v
    module.ALL_ATTENTION_FUNCTIONS = {"sdpa": attention}
    package = ModuleType("dflash")
    package.model = module
    monkeypatch.setitem(sys.modules, "dflash", package)
    monkeypatch.setitem(sys.modules, "dflash.model", module)
    draft = nn.Module()
    draft.block_size = 4
    layers = []
    for i in range(3):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.layer_idx = i
        layer.mlp = nn.Linear(2, 2)
        layers.append(layer)
    draft.layers = nn.ModuleList(layers)
    return draft, module, attention, calls


def test_removed_layers_reindex_cache_and_restore_after_failure(native_fixture):
    draft, _, _, _ = native_fixture
    original = draft.layers
    with pytest.raises(RuntimeError):
        with native_intervention(draft, {"kind": "drop_layers", "layers": [0]}):
            assert list(draft.layers) == list(original)[1:]
            assert [x.self_attn.layer_idx for x in draft.layers] == [0, 1]
            raise RuntimeError("probe failed")
    assert draft.layers is original
    assert [x.self_attn.layer_idx for x in original] == [0, 1, 2]


def test_attention_window_only_changes_drafter_and_preserves_sinks(native_fixture):
    draft, module, original, calls = native_fixture
    key = torch.arange(20.0).reshape(1, 1, 20, 1)
    with native_intervention(draft, {"kind": "window", "window": 4}):
        fn = module.ALL_ATTENTION_FUNCTIONS["sdpa"]
        fn(draft.layers[0].self_attn, key, key, key, None)
        assert calls[-1].flatten().tolist() == [0, 1, 2, 3, 16, 17, 18, 19]
        fn(nn.Module(), key, key, key, None)
        assert torch.equal(calls[-1], key)
    assert module.ALL_ATTENTION_FUNCTIONS["sdpa"] is original


def test_mlp_replacement_restores_original(native_fixture):
    draft, _, _, _ = native_fixture
    original = draft.layers[1].mlp
    with native_intervention(draft, {"kind": "zero_mlp", "layers": [1]}):
        assert torch.equal(draft.layers[1].mlp(torch.ones(2)), torch.zeros(2))
    assert draft.layers[1].mlp is original


def test_linear_fit_generalizes_to_unseen_inputs():
    torch.manual_seed(7)
    x, test = torch.randn(128, 12), torch.randn(32, 12)
    weight, bias = torch.randn(12, 8), torch.randn(8)
    fitted, loss = fit_linear_mlp(x, x @ weight + bias, 12, ridge=1e-6)
    error = (fitted(test.bfloat16()).float() - (test @ weight + bias)).square().mean()
    assert loss < 1e-8
    assert error < 0.002


def test_full_neuron_selection_preserves_nonlinear_mlp():
    torch.manual_seed(11)
    original = nn.Module()
    original.gate_proj = nn.Linear(8, 16, bias=False).bfloat16()
    original.up_proj = nn.Linear(8, 16, bias=False).bfloat16()
    original.down_proj = nn.Linear(16, 8, bias=False).bfloat16()
    original.act_fn = torch.nn.functional.silu
    x = torch.randn(32, 8).bfloat16()
    y = original.down_proj(original.act_fn(original.gate_proj(x))*original.up_proj(x))
    fitted, indices = fit_pruned_mlp(original, x.float(), y.float(), 1.0, 0)
    assert torch.equal(indices, torch.arange(16))
    assert torch.equal(fitted(x), y)


def test_shared_context_matches_separate_projections_and_restores(native_fixture):
    draft, _, _, _ = native_fixture
    torch.manual_seed(19)
    weights, originals = [], []
    for layer in draft.layers:
        for name in ["k_proj", "v_proj"]:
            projection = nn.Linear(8, 4, bias=False)
            setattr(layer.self_attn, name, projection)
            weights.append(projection.weight.detach())
            originals.append(projection.forward)
        def forward(*, target_hidden, hidden_states, layer=layer):
            return [(getattr(layer.self_attn, name)(target_hidden),
                     getattr(layer.self_attn, name)(hidden_states))
                    for name in ["k_proj", "v_proj"]]
        layer.forward = forward
    context, noise = torch.randn(1, 7, 8), torch.randn(1, 3, 8)
    baseline = [layer(target_hidden=context, hidden_states=noise) for layer in draft.layers]
    projector = FullLinear(torch.cat(weights))
    calls = []
    handle = projector.register_forward_hook(lambda *args: calls.append(1))
    with pytest.raises(RuntimeError):
        with native_intervention(draft, {"kind": "shared_context"}, {("context", 0): (projector, [4]*6)}):
            actual = [layer(target_hidden=context, hidden_states=noise) for layer in draft.layers]
            for expected_layer, actual_layer in zip(baseline, actual):
                for expected_pair, actual_pair in zip(expected_layer, actual_layer):
                    for expected, value in zip(expected_pair, actual_pair):
                        torch.testing.assert_close(value, expected)
            assert len(calls) == 1
            raise RuntimeError("restore probe")
    handle.remove()
    assert [getattr(layer.self_attn, name).forward for layer in draft.layers for name in ["k_proj", "v_proj"]] == originals


def test_folded_reconstruction_recovers_redundant_neurons_without_extra_layers():
    torch.manual_seed(29)
    original = nn.Module()
    original.gate_proj = nn.Linear(4, 16, bias=False).bfloat16()
    original.up_proj = nn.Linear(4, 16, bias=False).bfloat16()
    original.down_proj = nn.Linear(16, 4, bias=False).bfloat16()
    original.act_fn = torch.nn.functional.silu
    with torch.no_grad():
        original.gate_proj.weight.copy_(original.gate_proj.weight[0].clone().expand(16, -1))
        original.up_proj.weight.copy_(original.up_proj.weight[0].clone().expand(16, -1))
        x, heldout = torch.randn(128, 4).bfloat16(), torch.randn(64, 4).bfloat16()
        def reference(x):
            return original.down_proj(original.act_fn(original.gate_proj(x))*original.up_proj(x))
        fitted, indices = fit_pruned_mlp(original, x.float(), reference(x).float(), .5, 1, folded=True)
        error = (fitted(heldout).float()-reference(heldout).float()).square().sum()/reference(heldout).float().square().sum()
    assert len(indices) == 8
    assert fitted.correction is None
    assert len(list(fitted.children())) == 0
    assert error < .005


def test_prefix_refinement_rolls_back_draft_cache_and_preserves_anchors(native_fixture):
    draft, _, _, _ = native_fixture
    class Cache:
        length = 7
        def get_seq_length(self):
            return self.length
        def crop(self, n):
            self.length = n
    cache = Cache()
    calls = []
    first = torch.arange(12.0).reshape(1, 4, 3)
    def forward(*, noise_embedding, past_key_values):
        calls.append((past_key_values.length, noise_embedding.clone()))
        past_key_values.length += 9
        return first if len(calls) == 1 else first+100
    draft.forward = forward
    embedding = nn.Embedding(3, 3)
    noise = torch.zeros(1, 4, 3)
    with native_intervention(draft, {"kind": "prefix_refine", "prefix": 1}, {"native_head": nn.Identity(), "native_embedding": embedding}):
        out = draft(noise_embedding=noise, past_key_values=cache)
        assert [c[0] for c in calls] == [7, 7]
        assert cache.length == 16
        assert torch.equal(out[:, :2], first[:, :2])
        assert torch.equal(out[:, 2:], first[:, 2:]+100)
        assert torch.equal(calls[1][1][:, 0], noise[:, 0])
        assert torch.equal(calls[1][1][:, 2:], noise[:, 2:])
        torch.testing.assert_close(calls[1][1][:, 1], embedding(torch.tensor([2])))
    assert draft.forward is forward
