"""Layerwise source alignment and frozen fused-context supervision."""

import torch
from torch import nn


class LayerContextMapper(nn.Module):
    def __init__(self, layers, target_width, source_width):
        super().__init__()
        self.maps = nn.ModuleList(
            [nn.Linear(target_width, source_width, bias=False) for _ in range(layers)]
        )

    def forward(self, x):
        return torch.stack([m(x[:, i]) for i, m in enumerate(self.maps)], dim=1)

    def fold(self, fusion):
        width = self.maps[0].out_features
        return torch.cat(
            [
                fusion[:, i * width : (i + 1) * width].float() @ m.weight.float()
                for i, m in enumerate(self.maps)
            ],
            dim=1,
        )


def frozen_norm(x, gamma, eps=1e-6):
    # Preserve released RMSNorm convention: FP32 variance, input-dtype output.
    dtype = x.dtype
    z = x.float()
    z = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps)
    return gamma * z.to(dtype)


def relative_error(predicted, target):
    p, t = predicted.float(), target.detach().float()
    return ((p - t).square().sum(-1) / (t.square().sum(-1) + 1e-6)).mean()


def layer_context_loss(predicted_layers, target_layers, fusion, gamma):
    layer = relative_error(predicted_layers, target_layers)
    predicted = frozen_norm(
        torch.nn.functional.linear(predicted_layers.flatten(1), fusion), gamma
    )
    with torch.no_grad():
        target = frozen_norm(
            torch.nn.functional.linear(target_layers.flatten(1), fusion), gamma
        )
    context = relative_error(predicted, target)
    return layer, context
