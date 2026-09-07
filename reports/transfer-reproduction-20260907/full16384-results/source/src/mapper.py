"""Exactly the trained five-map added-linear context objective."""
import torch
from torch import nn
from torch.nn import functional as F
EPS=1e-6
def rms(x):return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+EPS)).to(x.dtype)
def relative(x,y):return (x.float()-y.float()).square().sum(-1)/(y.float().square().sum(-1)+EPS)
class Projection(nn.Module):
 def __init__(self,di=4096,do=2560):
  super().__init__();self.weight=nn.Parameter(torch.empty(do,di));nn.init.xavier_uniform_(self.weight)
 def forward(self,x):return F.linear(x,self.weight)
class Context(nn.Module):
 def __init__(self,fusion,norm,d8=4096,d4=2560,layers=5):
  super().__init__();self.d8=d8;self.d4=d4;self.layers=layers
  self.register_buffer('fusion',fusion.detach().clone());self.register_buffer('norm',norm.detach().clone())
  self.maps=nn.ModuleList([Projection(d8,d4) for _ in range(layers)])
 def frozen_norm(self,x):return rms(x)*self.norm.to(x.dtype)
 def forward(self,x):
  z=torch.cat([m(a) for m,a in zip(self.maps,x.split(self.d8,dim=-1))],dim=-1)
  return self.frozen_norm(F.linear(z,self.fusion)),z
 def loss(self,x,y):
  c,z=self(x);target=self.frozen_norm(F.linear(y,self.fusion))
  return relative(c,target)+relative(z.reshape(-1,self.layers,self.d4),y.reshape(-1,self.layers,self.d4)).mean(-1)
 def folded(self):
  return torch.cat([self.fusion[:,i*self.d4:(i+1)*self.d4].float()@a.weight.float() for i,a in enumerate(self.maps)],dim=1)
