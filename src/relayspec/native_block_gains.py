"""Calibrate frozen native projection blocks and fold gains for deployment."""

import torch
from torch import nn
from torch.nn.utils import parametrize


class BlockGains(nn.Module):
    def __init__(self, blocks, width, *, device, dtype):
        super().__init__()
        self.width = width
        self.gains = nn.Parameter(torch.ones(blocks, device=device, dtype=dtype))

    def forward(self, weight):
        return weight * self.gains.repeat_interleave(self.width).unsqueeze(0)


def enable_native_block_gains(relay, blocks):
    projection = relay.projection
    if (
        relay.normalize_input
        or not isinstance(projection, nn.Linear)
        or parametrize.is_parametrized(projection)
        or type(blocks) is not int
        or blocks < 1
        or projection.in_features % blocks
    ):
        raise ValueError(
            "Block gains require an unnormalized dense native map and complete blocks"
        )
    for parameter in relay.parameters():
        parameter.requires_grad_(False)
    parametrize.register_parametrization(
        projection,
        "weight",
        BlockGains(
            blocks,
            projection.in_features // blocks,
            device=projection.weight.device,
            dtype=projection.weight.dtype,
        ),
    )


def folded_relay_state(relay):
    state = relay.state_dict()
    if parametrize.is_parametrized(relay.projection, "weight"):
        state = {
            k: v
            for k, v in state.items()
            if not k.startswith("projection.parametrizations.weight.")
        }
        state["projection.weight"] = relay.projection.weight.detach().clone()
    return state
