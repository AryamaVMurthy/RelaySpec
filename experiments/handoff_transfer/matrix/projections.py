"""Explicit adapter architectures; all expose their exact dense deployment map."""
import torch
from torch import nn
from torch.nn import functional as F

class FiveLowRankMaps(nn.Module):
    """W_i = frozen ZIP W_i + B_i A_i, with alpha/r = 1."""
    def __init__(self, fusion, weights, rank=56):
        super().__init__()
        if len(weights)!=5 or rank<=0:raise ValueError('Five maps and positive rank required')
        self.source_width,self.target_width=weights[0].shape
        if fusion.shape!=(self.source_width,5*self.source_width):raise ValueError('Fusion dimensions differ')
        self.register_buffer('fusion',fusion.detach().clone())
        self.register_buffer('base',torch.stack([w.detach().clone() for w in weights]))
        self.A=nn.ParameterList([nn.Parameter(torch.empty(rank,self.target_width)) for _ in range(5)])
        self.B=nn.ParameterList([nn.Parameter(torch.zeros(self.source_width,rank)) for _ in range(5)])
        for a in self.A:nn.init.kaiming_uniform_(a,a=5**.5)
    def forward(self,x):
        pieces=[]
        for i,part in enumerate(x.split(self.target_width,dim=-1)):
            pieces.append(F.linear(part,self.base[i])+F.linear(F.linear(part,self.A[i]),self.B[i]))
        return F.linear(torch.cat(pieces,dim=-1),self.fusion)
    def folded(self):
        return torch.cat([self.fusion[:,i*self.source_width:(i+1)*self.source_width].float() @
            (self.base[i].float()+self.B[i].float()@self.A[i].float()) for i in range(5)],dim=1)

class DenseFusion(nn.Module):
    """One unrestricted matrix from concatenated target taps to source context."""
    def __init__(self,weight,normalize_input=False,eps=1e-6):
        super().__init__();self.weight=nn.Parameter(weight.detach().float().clone())
        self.normalize_input=normalize_input;self.eps=eps
    def forward(self,x):
        if self.normalize_input:
            x=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.eps)).to(x.dtype)
        return F.linear(x,self.weight)
    def folded(self):return self.weight
