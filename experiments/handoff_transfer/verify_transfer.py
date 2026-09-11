"""Require a trained interface, frozen drafter and faithful folded export."""
import argparse
from model import *

def main(a):
    torch.manual_seed(42)
    model,cfg=build(a.kind)
    draft=model.draft_model
    out=R/'modules'/a.tag/a.kind
    summary=json.loads((out/'summary.json').read_text())
    assert summary['optimizer_steps']==a.steps
    parameters=torch.load(out/'final.pt',weights_only=True,map_location='cuda')
    expected={k for k,p in draft.named_parameters() if p.requires_grad}
    assert set(parameters)==expected
    assert any(not torch.equal(dict(draft.named_parameters())[k],v) for k,v in parameters.items())
    draft.load_state_dict(parameters,strict=False)
    dest=R/'exports'/a.tag/a.kind
    exported=load_file(str(dest/'model.safetensors'))
    base=load_file(str(Path(TRANSFER['base_export'])/'model.safetensors'))
    assert set(exported)==set(base)
    assert all(torch.equal(v,exported[k]) for k,v in base.items() if k!='fc.weight')
    if a.kind=='five_maps':folded=draft.fc.folded()
    else:folded=draft.fc.base_layer.weight.float()+draft.fc.get_delta_weight('default').float()
    folded=folded.detach().to(dtype=torch.bfloat16,device='cpu')
    error=float((folded.float()-exported['fc.weight'].float()).square().sum()/folded.float().square().sum().clamp_min(1e-12))
    assert error<1e-6
    assert (dest/'config.json').read_bytes()==(Path(TRANSFER['base_export'])/'config.json').read_bytes()
    put(out/'verification.json',{'status':'passed','trainable_names':sorted(expected),'folded_relative_mse':error,
         'frozen_non_fc_exact':True,'export_sha256':sha(dest/'model.safetensors'),'steps':a.steps})
    print('TRANSFER_VERIFIED',a.kind,a.tag,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=VARIANTS,required=True)
    p.add_argument('--tag',required=True);p.add_argument('--steps',type=int,required=True);main(p.parse_args())
