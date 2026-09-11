"""Five independent linear maps followed by the frozen native fusion."""
import torch
from torch import nn
from torch.nn import functional as F

class FiveMapProjection(nn.Module):
    def __init__(self, fusion, weights):
        super().__init__()
        assert len(weights)==5 and fusion.ndim==2
        self.source_width, self.target_width=weights[0].shape
        assert fusion.shape==(self.source_width,5*self.source_width)
        assert all(w.shape==weights[0].shape for w in weights)
        self.register_buffer('fusion',fusion.detach().clone())
        self.maps=nn.ModuleList()
        for weight in weights:
            layer=nn.Module()
            layer.register_parameter('weight',nn.Parameter(weight.detach().float().clone()))
            self.maps.append(layer)

    def forward(self, x):
        assert x.shape[-1]==5*self.target_width
        z=torch.cat([F.linear(a,m.weight) for a,m in zip(x.split(self.target_width,dim=-1),self.maps)],dim=-1)
        return F.linear(z,self.fusion)

    def folded(self):
        return torch.cat([self.fusion[:,i*self.source_width:(i+1)*self.source_width].float() @ m.weight.float()
                          for i,m in enumerate(self.maps)],dim=1)
