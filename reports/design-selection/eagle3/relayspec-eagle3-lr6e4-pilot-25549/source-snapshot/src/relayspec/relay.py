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
    ) -> None:
        super().__init__()
        if target_hidden_size <= 0 or num_taps <= 0 or draft_hidden_size <= 0:
            raise ValueError("relay dimensions must be positive")
        self.input_width = target_hidden_size * num_taps
        self.input_norm = nn.RMSNorm(
            self.input_width,
            eps=eps,
            elementwise_affine=False,
        )
        self.projection = nn.Linear(self.input_width, draft_hidden_size, bias=False)

    def forward(self, target_taps: torch.Tensor) -> torch.Tensor:
        if target_taps.ndim != 3:
            raise ValueError("relay input must have shape [batch, sequence, width]")
        if target_taps.shape[-1] != self.input_width:
            raise ValueError(
                f"relay input width must be {self.input_width}, "
                f"found {target_taps.shape[-1]}"
            )
        return self.projection(self.input_norm(target_taps))
