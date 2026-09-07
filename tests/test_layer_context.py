import torch

from relayspec.layer_context import (
    LayerContextMapper,
    layer_context_loss,
    relative_error,
)


def test_fold_matches_two_stage_and_frozen_teacher_has_no_gradient():
    torch.manual_seed(3)
    m = LayerContextMapper(5, 7, 4)
    f = torch.randn(4, 20)
    gamma = torch.randn(4)
    x = torch.randn(11, 5, 7)
    y = torch.randn(11, 5, 4)
    prediction = m(x)
    torch.testing.assert_close(prediction.flatten(1) @ f.T, x.flatten(1) @ m.fold(f).T)
    a, b = layer_context_loss(prediction, y, f, gamma)
    (a + b).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )
    assert f.grad is None and gamma.grad is None and y.grad is None


def test_relative_error_equal_layer_token_weighting_and_epsilon():
    p = torch.tensor([[[2.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [2.0, 0.0]]])
    t = torch.tensor([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]])
    torch.testing.assert_close(relative_error(p, t), torch.tensor(0.5 / (1 + 1e-6)))
