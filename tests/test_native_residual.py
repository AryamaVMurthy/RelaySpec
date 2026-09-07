import torch

from relayspec.native_block_gains import folded_relay_state
from relayspec.native_residual import enable_native_residual
from relayspec.relay import TargetFeatureRelay


def test_residual_preserves_initial_map_learns_and_exports_without_extra_operators():
    torch.manual_seed(1729)
    kwargs = dict(
        target_hidden_size=3,
        num_taps=2,
        draft_hidden_size=4,
        eps=1e-6,
        normalize_input=False,
    )
    relay = TargetFeatureRelay(**kwargs)
    x = torch.randn(2, 5, 6)
    original = relay.projection.weight.detach().clone()
    before = relay(x).detach().clone()
    enable_native_residual(relay, 2)
    torch.testing.assert_close(relay(x), before, rtol=0, atol=0)
    assert sum(p.numel() for p in relay.parameters() if p.requires_grad) == 20
    optimizer = torch.optim.SGD(relay.parameters(), lr=0.1)
    for _ in range(2):
        optimizer.zero_grad()
        relay(x).square().mean().backward()
        optimizer.step()
    correction = relay.projection.parametrizations.weight[0]
    assert correction.up.grad.abs().sum() > 0
    assert correction.down.grad.abs().sum() > 0
    assert relay.projection.parametrizations.weight.original.grad is None
    torch.testing.assert_close(
        relay.projection.parametrizations.weight.original, original
    )
    assert not torch.equal(relay(x), before)
    restored = TargetFeatureRelay(**kwargs)
    restored.load_state_dict(folded_relay_state(relay), strict=True)
    torch.testing.assert_close(restored(x), relay(x), rtol=0, atol=0)
