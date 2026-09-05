import io

import pytest
import torch

from relayspec.relay import TargetFeatureRelay


def make(**kwargs):
    return TargetFeatureRelay(
        target_hidden_size=4, num_taps=3, draft_hidden_size=5, eps=1e-6, **kwargs
    )


def test_factorized_map_matches_collapsed_matrix_and_checkpoint():
    relay = make(factorized_rank=3)
    x = torch.randn(2, 7, 12)
    weight = relay.projection[1].weight @ relay.projection[0].weight
    expected = torch.nn.functional.linear(relay.input_norm(x), weight)
    torch.testing.assert_close(relay(x), expected)
    buffer = io.BytesIO()
    torch.save(relay.state_dict(), buffer)
    buffer.seek(0)
    restored = make(factorized_rank=3)
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    torch.testing.assert_close(restored(x), expected)
    relay(x).square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in relay.parameters()
    )


def test_factorized_and_mlp_have_identical_parameter_budgets():
    for width in (2, 5, 16):
        linear = make(factorized_rank=width)
        mlp = make(mlp_hidden_width=width)
        assert sum(p.numel() for p in linear.parameters()) == width * (12 + 5)
        assert sum(p.numel() for p in mlp.parameters()) == width * (12 + 5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"factorized_rank": 0},
        {"factorized_rank": -1},
        {"factorized_rank": 2, "mlp_hidden_width": 2},
    ],
)
def test_invalid_capacity_is_rejected(kwargs):
    with pytest.raises(ValueError):
        make(**kwargs)


def test_explicit_l2_has_correct_gradient_and_excludes_frozen_parameters():
    from relayspec.losses import explicit_l2_penalty

    trained = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
    frozen = torch.nn.Parameter(torch.tensor([100.0]), requires_grad=False)
    penalty = explicit_l2_penalty([trained, frozen], 0.2)
    torch.testing.assert_close(penalty, torch.tensor(1.3))
    penalty.backward()
    torch.testing.assert_close(trained.grad, torch.tensor([0.4, -0.6]))
    assert frozen.grad is None
    assert explicit_l2_penalty([trained], 0.0).item() == 0.0
    for coefficient in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            explicit_l2_penalty([trained], coefficient)


def test_fitting_validation_checks_content_instead_of_supplied_hash():
    from relayspec.fitting_validation import validate_fit_split

    train = [{"problem": "a b", "solution": "s", "problem_sha256": "fake1"}]
    validation = [{"problem": "a  b", "solution": "s2", "problem_sha256": "fake2"}]
    with pytest.raises(ValueError, match="disjoint"):
        validate_fit_split(train, validation)


def test_fitting_diagnostics_use_record_weighting_and_preserve_gradients():
    from relayspec.fitting_validation import interface_diagnostics

    relay = make(normalize_input=False)
    x = torch.randn(1, 3, 12)
    y = relay(x).detach()
    result = interface_diagnostics(
        relay,
        [(x, y), (x[:, :1], y[:, :1])],
        device=torch.device("cpu"),
        objective="relative_interface_mse",
    )
    assert result["records"] == 2 and result["tokens"] == 4
    assert result["relative_mse"] < 1e-12
    assert all(p.grad is None for p in relay.parameters())
