"""Verify the unchanged package's two-update checkpoint before inference."""
import argparse,json,os
from pathlib import Path
import torch
from safetensors.torch import load_file
root=Path(os.environ['AUF_WORKDIR']);models=Path(os.environ['AUF_MODELS'])
p=argparse.ArgumentParser();p.add_argument('--tag',default='check');p.add_argument('--steps',type=int,default=2);p.add_argument('--records',type=int,default=32);args=p.parse_args()
summary=json.loads((root/f'modules/{args.tag}/fusion_r56/summary.json').read_text())
assert summary['optimizer_steps']==args.steps and summary['trainable_parameters']==860160
parameters=torch.load(root/f'modules/{args.tag}/fusion_r56/final.pt',weights_only=True,map_location='cpu')
assert set(parameters)=={'fc.lora_A.default.weight','fc.lora_B.default.weight'}
source=load_file(str(models/'draft/model.safetensors'))
export=load_file(str(root/f'exports/{args.tag}/fusion_r56/model.safetensors'))
assert source.keys()==export.keys()
assert all(torch.equal(source[k],export[k]) for k in source if k!='fc.weight')
assert not torch.equal(source['fc.weight'],export['fc.weight'])
expected=(source['fc.weight'].float()+parameters['fc.lora_B.default.weight'].float()@parameters['fc.lora_A.default.weight'].float()).bfloat16()
relative=((expected.float()-export['fc.weight'].float()).square().sum()/expected.float().square().sum()).item()
assert relative<1e-6,relative
report={'status':'passed','steps':args.steps,'unique_cached_records':args.records,'trained_parameters':860160,
        'all_non_fusion_weights_exact':True,'folded_export_relative_mse':relative,
        'interpretation':'Weight/export check only; speed and full-output checks are separate'}
(root/'gate-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
