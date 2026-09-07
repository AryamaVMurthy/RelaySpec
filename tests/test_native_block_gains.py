import torch

from relayspec.native_block_gains import enable_native_block_gains, folded_relay_state
from relayspec.relay import TargetFeatureRelay


def test_only_layer_gains_train_and_fold_into_standard_checkpoint():
    relay = TargetFeatureRelay(
        target_hidden_size=3,
        num_taps=2,
        draft_hidden_size=4,
        eps=1e-6,
        normalize_input=False,
    )
    original = relay.projection.weight.detach().clone()
    enable_native_block_gains(relay, 2)
    assert sum(p.numel() for p in relay.parameters() if p.requires_grad) == 2
    gains = relay.projection.parametrizations.weight[0].gains
    with torch.no_grad():
        gains.copy_(torch.tensor([0.7, 1.3]))
    x = torch.randn(2, 5, 6)
    expected = torch.nn.functional.linear(
        x, original * torch.tensor([0.7] * 3 + [1.3] * 3)
    )
    torch.testing.assert_close(relay(x), expected)
    relay(x).square().mean().backward()
    assert gains.grad is not None and torch.isfinite(gains.grad).all()
    assert relay.projection.parametrizations.weight.original.grad is None
    restored = TargetFeatureRelay(
        target_hidden_size=3,
        num_taps=2,
        draft_hidden_size=4,
        eps=1e-6,
        normalize_input=False,
    )
    restored.load_state_dict(folded_relay_state(relay), strict=True)
    torch.testing.assert_close(restored(x), relay(x))
    torch.testing.assert_close(
        relay.projection.parametrizations.weight.original, original
    )
