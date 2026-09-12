"""The original normalized linear relay with its frozen source output norm."""
import torch
from torch import nn
from torch.nn import functional as F
from relayspec.relay import TargetFeatureRelay

class NormalInterface(nn.Module):
    def __init__(self, fusion, norm, target_width, input_eps, output_eps):
        super().__init__()
        self.relay=TargetFeatureRelay(target_hidden_size=target_width,num_taps=5,
            draft_hidden_size=fusion.shape[0],eps=input_eps,normalize_input=True)
        self.register_buffer('fusion',fusion.detach().clone())
        self.register_buffer('norm',norm.detach().clone())
        self.output_eps=output_eps

    def normalize(self,x):
        return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.output_eps)).to(x.dtype)*self.norm.to(x.dtype)

    def forward(self,x):
        return self.normalize(self.relay(x.unsqueeze(0)).squeeze(0))

    def feature_loss(self,x,y,objective="relative_mse"):
        target=self.normalize(F.linear(y,self.fusion)).detach()
        prediction=self(x)
        if objective != "relative_mse":
            from experiments.handoff_transfer.feature_objectives import feature_distribution_loss
            return feature_distribution_loss(prediction,target,objective)
        # Original relative_interface_mse's per-token denominator and reduction,
        # kept unreduced here to preserve the ZIP sampler's equal-record mass.
        return (prediction.float()-target.float()).square().sum(-1)/target.float().square().sum(-1).clamp_min(torch.finfo(torch.float32).tiny)
