"""Fit a low-rank correction to inherited native columns, then fold for export."""

from torch import nn
from torch.nn.utils import parametrize


class NativeResidual(nn.Module):
    def __init__(self, weight, rank):
        super().__init__()
        self.down = nn.Parameter(weight.new_empty(rank, weight.shape[1]))
        self.up = nn.Parameter(weight.new_zeros(weight.shape[0], rank))
        nn.init.normal_(self.down, std=weight.shape[1] ** -0.5)

    def forward(self, weight):
        return weight + self.up @ self.down


def enable_native_residual(relay, rank):
    projection = relay.projection
    if (
        relay.normalize_input
        or not isinstance(projection, nn.Linear)
        or parametrize.is_parametrized(projection)
        or type(rank) is not int
        or not 1 <= rank <= min(projection.in_features, projection.out_features)
    ):
        raise ValueError(
            "Native residual requires an unnormalized dense map and valid rank"
        )
    for parameter in relay.parameters():
        parameter.requires_grad_(False)
    parametrize.register_parametrization(
        projection, "weight", NativeResidual(projection.weight, rank)
    )
