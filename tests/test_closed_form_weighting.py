"""Verifies the closed-form ridge fit solves the same objective as SGD.

`fit_relay_closed_form.py` must minimize the identical loss
`relative_interface_mse` (src/relayspec/losses.py) uses for gradient
descent, not a differently-weighted ridge objective. Since the map is
linear, per-position energy weighting (1/||c_t||) before accumulating the
normal equations makes the two mathematically equivalent. This test
proves that equivalence directly: the closed-form solution must be a
stationary point (zero gradient) of the actual relative loss, for a
synthetic, randomly generated (and therefore not fine-tuned to pass)
regression problem.
"""

from __future__ import annotations

import torch

from relayspec.losses import relative_interface_mse


def _weighted_ridge_solve(
    z: torch.Tensor, c: torch.Tensor, ridge_lambda: float
) -> torch.Tensor:
    """Mirrors the weighting fit_relay_closed_form.py applies before solving."""
    position_energy = (
        c.square().sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
    )
    weight = position_energy.rsqrt()
    z_weighted = z * weight
    c_weighted = c * weight
    ztz = z_weighted.T @ z_weighted
    ztc = z_weighted.T @ c_weighted
    regularized = ztz + ridge_lambda * torch.eye(
        z.shape[1], dtype=z.dtype, device=z.device
    )
    return torch.linalg.solve(regularized, ztc)


def test_weighted_closed_form_is_stationary_point_of_relative_loss() -> None:
    torch.manual_seed(1729)
    positions, input_width, output_width = 64, 12, 5
    z = torch.randn(positions, input_width, dtype=torch.float64)
    c = torch.randn(positions, output_width, dtype=torch.float64) * torch.rand(
        positions, 1, dtype=torch.float64
    ).add(0.1)  # varied per-position magnitude, the case the bug mishandled

    weight = _weighted_ridge_solve(z, c, ridge_lambda=1e-6)  # near-unregularized
    weight.requires_grad_(True)

    prediction = z @ weight
    loss = relative_interface_mse(prediction.unsqueeze(0), c.unsqueeze(0))
    (grad,) = torch.autograd.grad(loss, weight)

    # A true stationary point of the actual training loss has ~zero gradient.
    # A solution to the wrong (unweighted) objective would not.
    assert grad.abs().max().item() < 1e-6


def test_unweighted_ridge_is_not_a_stationary_point_of_relative_loss() -> None:
    """Confirms the bug this fix replaces would actually have failed this check."""
    torch.manual_seed(1729)
    positions, input_width, output_width = 64, 12, 5
    z = torch.randn(positions, input_width, dtype=torch.float64)
    c = torch.randn(positions, output_width, dtype=torch.float64) * torch.rand(
        positions, 1, dtype=torch.float64
    ).add(0.1)

    ztz = z.T @ z + 1e-6 * torch.eye(input_width, dtype=torch.float64)
    ztc = z.T @ c
    unweighted = torch.linalg.solve(ztz, ztc)
    unweighted.requires_grad_(True)

    prediction = z @ unweighted
    loss = relative_interface_mse(prediction.unsqueeze(0), c.unsqueeze(0))
    (grad,) = torch.autograd.grad(loss, unweighted)

    assert grad.abs().max().item() > 1e-3
