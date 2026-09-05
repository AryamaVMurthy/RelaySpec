from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_extract_hidden_taps_uses_post_layer_states() -> None:
    from relayspec.relay import extract_hidden_taps

    hidden_states = tuple(torch.full((1, 3, 2), float(index)) for index in range(6))
    selected = extract_hidden_taps(hidden_states, (0, 2, 4))

    assert selected.shape == (1, 3, 6)
    assert selected[0, 0].tolist() == [1.0, 1.0, 3.0, 3.0, 5.0, 5.0]


def test_target_feature_relay_projects_concatenated_taps() -> None:
    from relayspec.relay import TargetFeatureRelay

    relay = TargetFeatureRelay(
        target_hidden_size=4,
        num_taps=3,
        draft_hidden_size=5,
        eps=1e-6,
    )
    assert relay(torch.randn(2, 7, 12)).shape == (2, 7, 5)


def test_target_feature_relay_rejects_wrong_width() -> None:
    from relayspec.relay import TargetFeatureRelay

    relay = TargetFeatureRelay(
        target_hidden_size=4,
        num_taps=3,
        draft_hidden_size=5,
        eps=1e-6,
    )
    with pytest.raises(ValueError, match="relay input width"):
        relay(torch.randn(1, 2, 11))


def test_target_feature_relay_adapter_maps_different_input_width() -> None:
    from relayspec.relay import TargetFeatureRelay

    relay = TargetFeatureRelay(
        target_hidden_size=4,
        num_taps=3,
        draft_hidden_size=5,
        eps=1e-6,
        adapter_input_width=20,
    )
    assert relay(torch.randn(2, 7, 20)).shape == (2, 7, 5)
    with pytest.raises(ValueError, match="relay input width must be 20"):
        relay(torch.randn(1, 2, 12))


def test_target_feature_relay_adapter_freezes_base_independently() -> None:
    from relayspec.relay import TargetFeatureRelay

    relay = TargetFeatureRelay(
        target_hidden_size=4,
        num_taps=3,
        draft_hidden_size=5,
        eps=1e-6,
        adapter_input_width=20,
    )
    relay.projection.weight.requires_grad_(False)
    trainable = [name for name, p in relay.named_parameters() if p.requires_grad]
    assert trainable == ["adapter.weight"]


def test_target_feature_relay_delta_starts_as_a_no_op() -> None:
    from relayspec.relay import TargetFeatureRelay

    torch.manual_seed(0)
    base = TargetFeatureRelay(
        target_hidden_size=4, num_taps=3, draft_hidden_size=5, eps=1e-6
    )
    delta_relay = TargetFeatureRelay(
        target_hidden_size=4,
        num_taps=3,
        draft_hidden_size=5,
        eps=1e-6,
        delta_rank=2,
    )
    delta_relay.load_state_dict(base.state_dict(), strict=False)

    inputs = torch.randn(2, 7, 12)
    assert torch.allclose(base(inputs), delta_relay(inputs))


def test_target_feature_relay_delta_freezes_base_independently() -> None:
    from relayspec.relay import TargetFeatureRelay

    relay = TargetFeatureRelay(
        target_hidden_size=4,
        num_taps=3,
        draft_hidden_size=5,
        eps=1e-6,
        delta_rank=2,
    )
    relay.projection.weight.requires_grad_(False)
    trainable = sorted(name for name, p in relay.named_parameters() if p.requires_grad)
    assert trainable == ["delta_down.weight", "delta_up.weight"]


def test_target_feature_relay_nonlinear_delta_also_starts_as_a_no_op() -> None:
    """c = R z + U sigma(V z): zero-initialized U makes this a no-op too,
    same as the linear delta, since sigma(V z) is scaled by zero either way.
    """
    from relayspec.relay import TargetFeatureRelay

    torch.manual_seed(0)
    base = TargetFeatureRelay(
        target_hidden_size=4, num_taps=3, draft_hidden_size=5, eps=1e-6
    )
    nonlinear_delta_relay = TargetFeatureRelay(
        target_hidden_size=4,
        num_taps=3,
        draft_hidden_size=5,
        eps=1e-6,
        delta_rank=2,
        delta_nonlinear=True,
    )
    nonlinear_delta_relay.load_state_dict(base.state_dict(), strict=False)

    inputs = torch.randn(2, 7, 12)
    assert torch.allclose(base(inputs), nonlinear_delta_relay(inputs))


def test_target_feature_relay_nonlinear_delta_actually_applies_gelu() -> None:
    """Once delta_up is non-zero, the nonlinear and linear delta paths must
    differ (this would fail if the GELU were silently skipped)."""
    from relayspec.relay import TargetFeatureRelay

    torch.manual_seed(0)
    linear_delta = TargetFeatureRelay(
        target_hidden_size=4,
        num_taps=3,
        draft_hidden_size=5,
        eps=1e-6,
        delta_rank=2,
        delta_nonlinear=False,
    )
    nonlinear_delta = TargetFeatureRelay(
        target_hidden_size=4,
        num_taps=3,
        draft_hidden_size=5,
        eps=1e-6,
        delta_rank=2,
        delta_nonlinear=True,
    )
    nonlinear_delta.load_state_dict(linear_delta.state_dict(), strict=True)
    torch.nn.init.normal_(linear_delta.delta_up.weight)
    torch.nn.init.normal_(nonlinear_delta.delta_up.weight)
    nonlinear_delta.delta_up.weight.data.copy_(linear_delta.delta_up.weight.data)

    inputs = torch.randn(2, 7, 12)
    assert not torch.allclose(linear_delta(inputs), nonlinear_delta(inputs))


def test_target_feature_relay_delta_nonlinear_requires_delta_rank() -> None:
    from relayspec.relay import TargetFeatureRelay

    with pytest.raises(ValueError, match="delta_nonlinear requires delta_rank"):
        TargetFeatureRelay(
            target_hidden_size=4,
            num_taps=3,
            draft_hidden_size=5,
            eps=1e-6,
            delta_nonlinear=True,
        )
