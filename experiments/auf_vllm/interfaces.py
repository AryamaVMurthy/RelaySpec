"""ZIP-compatible five-map interface and optional frozen-base fusion LoRA."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def relative_error(prediction, target):
    return (prediction.float()-target.float()).square().sum(-1) / (target.float().square().sum(-1)+1e-6)


class LayerContextMapper(nn.Module):
    def __init__(self, fusion, norm, target_width=4096, source_width=2560, num_taps=5):
        super().__init__()
        if fusion.shape != (source_width, source_width*num_taps) or norm.shape != (source_width,):
            raise ValueError("Fusion/norm shape does not match declared source interface")
        self.target_width = target_width
        self.source_width = source_width
        self.num_taps = num_taps
        self.register_buffer("fusion", fusion.detach().clone())
        self.register_buffer("norm", norm.detach().clone())
        # Allocate without nn.Linear's extra RNG draws: preserve ZIP's Xavier stream.
        self.maps = nn.ModuleList()
        for _ in range(num_taps):
            layer = nn.Module()
            layer.register_parameter("weight", nn.Parameter(torch.empty(source_width, target_width)))
            nn.init.xavier_uniform_(layer.weight)
            self.maps.append(layer)

    def normalize(self, x):
        normalized = (x.float()*torch.rsqrt(x.float().square().mean(-1, keepdim=True)+1e-6)).to(x.dtype)
        return normalized*self.norm.to(x.dtype)

    def forward(self, x):
        if x.shape[-1] != self.target_width*self.num_taps:
            raise ValueError("Wrong target feature width")
        z = torch.cat([F.linear(a, m.weight) for m, a in zip(self.maps, x.split(self.target_width, dim=-1))], -1)
        return self.normalize(F.linear(z, self.fusion)), z

    def feature_loss(self, x, y):
        if y.shape != (*x.shape[:-1], self.source_width*self.num_taps):
            raise ValueError("Paired source features have the wrong shape")
        context, z = self(x)
        target = self.normalize(F.linear(y, self.fusion))
        shape = (*x.shape[:-1], self.num_taps, self.source_width)
        return relative_error(context, target) + relative_error(z.reshape(shape), y.reshape(shape)).mean(-1)

    def folded(self):
        return torch.cat([self.fusion[:, i*self.source_width:(i+1)*self.source_width].float() @ layer.weight.float()
                          for i, layer in enumerate(self.maps)], dim=1)


class FusionLoRA(nn.Module):
    def __init__(self, base, rank=56, alpha=56):
        super().__init__()
        if base.ndim != 2 or rank <= 0 or alpha <= 0:
            raise ValueError("Invalid fusion or LoRA dimensions")
        self.register_buffer("base", base.detach().clone())
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.empty(rank, base.shape[1], dtype=torch.float32, device=base.device))
        self.B = nn.Parameter(torch.zeros(base.shape[0], rank, dtype=torch.float32, device=base.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(x, self.base) + (self.alpha/self.rank)*F.linear(F.linear(x, self.A), self.B)

    def folded(self):
        return self.base.float() + (self.alpha/self.rank)*(self.B @ self.A)
