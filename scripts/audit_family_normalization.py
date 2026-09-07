"""Quantify the fitting/runtime input-epsilon difference at fixed features."""
import json
import os
from pathlib import Path
import torch

torch.backends.cuda.matmul.allow_tf32=False
root=Path(os.environ['FAMILY_SCALE_CACHE']);results=[]
for pair,relative in [('llama','n16384-llama-dense/epoch-6.pt'),('cross','refine-cross-long/epoch-3.pt')]:
    meta=torch.load(root/pair/'metadata.pt',weights_only=True)
    state=torch.load(root/'fits'/relative,weights_only=True)
    weight=state['relay']['projection.weight'].cuda().float()
    gamma=meta['norm'].cuda().to(torch.bfloat16)
    values=[];exact=0;positions=0
    with torch.inference_mode():
        for i in range(256):
            x=torch.load(root/pair/'validation'/f'{i:05d}.pt',weights_only=True,mmap=True)['x'].cuda().float()
            outputs=[]
            for eps in (1e-6,1e-5):
                u=torch.nn.functional.rms_norm(x,(x.shape[-1],),eps=eps)
                u=torch.nn.functional.linear(u,weight).to(torch.bfloat16)
                y=(u.float()*torch.rsqrt(u.float().square().mean(-1,keepdim=True)+1e-6)).to(u.dtype)*gamma
                outputs.append(y)
            a,b=outputs
            v=(a.float()-b.float()).square().sum(-1)/(a.float().square().sum(-1)+1e-6)
            values.extend(v.cpu().tolist());positions+=len(x)
            exact+=int((a==b).all(-1).sum())
    results.append({'pair':pair,'fitting_input_epsilon':1e-6,'runtime_input_epsilon':1e-5,
        'output_epsilon':1e-6,'validation_records':256,'positions':positions,
        'mean_context_relative_mse':sum(values)/len(values),'max_context_relative_mse':max(values),
        'exact_context_vectors':exact,'scope':'Fixed cached BF16 features promoted to FP32; isolates epsilon only. Runtime configuration and weights were not changed.'})
(root/'normalization-audit.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
