"""Synthetic GPU correctness probe only; never exports a trained candidate."""
import json,sys
from pathlib import Path
import torch
sys.path.insert(0,'src')
from train import make
from paths import WORK
m=make().cuda();assert sum(p.numel() for p in m.parameters())==52428800
assert not m.fusion.requires_grad and not m.norm.requires_grad
torch.manual_seed(42);x=torch.randn(128,20480,device='cuda',dtype=torch.bfloat16);y=torch.randn(128,12800,device='cuda',dtype=torch.bfloat16)
with torch.autocast('cuda',dtype=torch.bfloat16):loss=m.loss(x,y).mean()
assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
with torch.no_grad():
 z=torch.cat([a(v) for a,v in zip(m.maps,x.float().split(4096,dim=-1))],-1)
 a=z@m.fusion.float().T;b=x.float()@m.folded().T
 error=float((a-b).square().sum()/a.square().sum());assert error<1e-8
p=WORK/'validation/package_mapper_gpu.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(dict(passed=True,synthetic=True,trained_candidate=False,parameters=52428800,fold_relative_error=error,loss=float(loss.detach()),device=torch.cuda.get_device_name(0)),indent=2)+'\n')
print(p.read_text())
