from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def extract_hidden_taps(
    hidden_states: Sequence[torch.Tensor],
    layer_ids: Sequence[int],
) -> torch.Tensor:
    """Concatenate post-layer hidden states using Transformers' embedding offset."""
    taps = tuple(int(layer_id) for layer_id in layer_ids)
    if not taps or tuple(sorted(set(taps))) != taps:
        raise ValueError("relay layer ids must be sorted and unique")
    if taps[0] < 0 or taps[-1] + 1 >= len(hidden_states):
        raise ValueError("relay layer id lies outside returned hidden states")
    return torch.cat([hidden_states[layer_id + 1] for layer_id in taps], dim=-1)


class TargetFeatureRelay(nn.Module):
    """Map cached target taps directly into DFlash's conditioned hidden space."""

    def __init__(
        self,
        *,
        target_hidden_size: int,
        num_taps: int,
        draft_hidden_size: int,
        eps: float,
        normalize_input: bool = True,
        adapter_input_width: int | None = None,
        delta_rank: int | None = None,
        delta_nonlinear: bool = False,
        mlp_hidden_width: int | None = None,
    ) -> None:
        super().__init__()
        if target_hidden_size <= 0 or num_taps <= 0 or draft_hidden_size <= 0:
            raise ValueError("relay dimensions must be positive")
        self.input_width = target_hidden_size * num_taps
        self.normalize_input = bool(normalize_input)
        self.input_norm = nn.RMSNorm(
            self.input_width,
            eps=eps,
            elementwise_affine=False,
        )
        # Capacity/architecture ablation (rigor pass): is a single linear
        # map the right hypothesis class, or does a small nonlinear map do
        # better at the same fitting budget? `mlp_hidden_width` swaps the
        # single `nn.Linear` for a two-layer GELU MLP of the given hidden
        # width; None (the default, and every result before this ablation)
        # keeps the original single linear map exactly as before.
        self.mlp_hidden_width = (
            int(mlp_hidden_width) if mlp_hidden_width is not None else None
        )
        if self.mlp_hidden_width is not None:
            if self.mlp_hidden_width <= 0:
                raise ValueError("mlp hidden width must be positive")
            self.projection = nn.Sequential(
                nn.Linear(self.input_width, self.mlp_hidden_width, bias=False),
                nn.GELU(),
                nn.Linear(self.mlp_hidden_width, draft_hidden_size, bias=False),
            )
        else:
            self.projection = nn.Linear(self.input_width, draft_hidden_size, bias=False)
        # E4-P1: an optional adapter that maps a different target's raw tap
        # width into this relay's native input width, so a frozen relay
        # fit for one target can be tested against another target's hidden
        # states. See docs/plans/2026-09-02-relayspec-source-free-adaptive-relay-plan.md
        # section 3.5, P1.
        self.adapter_input_width = (
            int(adapter_input_width) if adapter_input_width is not None else None
        )
        self.adapter: nn.Linear | None = None
        if self.adapter_input_width is not None:
            if self.adapter_input_width <= 0:
                raise ValueError("adapter input width must be positive")
            self.adapter = nn.Linear(
                self.adapter_input_width, self.input_width, bias=False
            )
        # E3 Stage B: an optional low-rank delta dR = delta_up(delta_down(x))
        # added to a frozen base projection, for adapting an already-fit
        # relay to a fine-tuned descendant target with identical widths. See
        # docs/plans/2026-09-02-relayspec-source-free-adaptive-relay-plan.md
        # section 3.4. `delta_up` starts at zero so an unfit delta is a
        # no-op, matching the standard low-rank-adapter convention.
        self.delta_rank = int(delta_rank) if delta_rank is not None else None
        self.delta_down: nn.Linear | None = None
        self.delta_up: nn.Linear | None = None
        # Residual nonlinear correction ablation: c = R z + U sigma(V z).
        # `delta_nonlinear=True` inserts a GELU between the existing
        # low-rank delta_down (V) and delta_up (U), turning the E3-Stage-B
        # linear delta into a small nonlinear residual path. R (the base
        # `projection`) is untouched and is expected to be loaded frozen
        # from an already-fit checkpoint; delta_up is zero-initialized
        # exactly as before, so the nonlinear path starts as an exact
        # no-op and can only ever learn a residual on top of the linear
        # solution, never replace it (unlike the separate `mlp_hidden_width`
        # ablation, which swaps R out for an MLP entirely).
        self.delta_nonlinear = bool(delta_nonlinear)
        self.delta_activation = nn.GELU() if self.delta_nonlinear else None
        if self.delta_rank is not None:
            if self.delta_rank <= 0:
                raise ValueError("delta rank must be positive")
            self.delta_down = nn.Linear(self.input_width, self.delta_rank, bias=False)
            self.delta_up = nn.Linear(self.delta_rank, draft_hidden_size, bias=False)
            nn.init.zeros_(self.delta_up.weight)
        elif self.delta_nonlinear:
            raise ValueError("delta_nonlinear requires delta_rank to be set")

    def forward(self, target_taps: torch.Tensor) -> torch.Tensor:
        if target_taps.ndim != 3:
            raise ValueError("relay input must have shape [batch, sequence, width]")
        expected_width = (
            self.adapter_input_width if self.adapter is not None else self.input_width
        )
        if target_taps.shape[-1] != expected_width:
            raise ValueError(
                f"relay input width must be {expected_width}, "
                f"found {target_taps.shape[-1]}"
            )
        relay_input = self.adapter(target_taps) if self.adapter is not None else target_taps
        relay_input = (
            self.input_norm(relay_input) if self.normalize_input else relay_input
        )
        output = self.projection(relay_input)
        if self.delta_down is not None and self.delta_up is not None:
            bottleneck = self.delta_down(relay_input)
            if self.delta_activation is not None:
                bottleneck = self.delta_activation(bottleneck)
            output = output + self.delta_up(bottleneck)
        return output
