"""Fit the original layer+context objective on full cross-family paired features."""
import argparse,json,math,time
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from experiments.auf_vllm.interfaces import LayerContextMapper
from experiments.auf_vllm.pilot_data import write
from experiments.handoff_transfer.matrix.prepare import digest
from experiments.handoff_transfer.cross.paired_batches import paired_index,batches


def main(args):
    items=paired_index(args.index)
    assert len(items)==4096,'A full4096-record initializer is required'
    reference=json.loads((args.reference/'transfer.json').read_text())
    base=Path(reference['base_export']);native=Path(reference['draft'])
    assert digest(base/'model.safetensors')==reference['base_sha256']
    weights=load_file(str(native/'model.safetensors'))
    torch.manual_seed(42)
    mapper=LayerContextMapper(weights['fc.weight'],weights['hidden_norm.weight']).cuda()
    assert sum(p.numel() for p in mapper.parameters())==52428800
    optimizer=torch.optim.AdamW(mapper.parameters(),lr=.001,weight_decay=0,fused=True)
    total=sum(r['positions'] for r in items);steps_epoch=math.ceil(total/2048);steps=3*steps_epoch
    warm=max(1,int(.05*steps));history=[];step=0;first_epoch=0
    contract=dict(records=4096,epochs=3,lr=.001,seed=42,position_batch=2048,
        index_sha256=digest(args.index),paired_files_sha256={r['path']:r['sha256'] for r in items},
        base_sha256=reference['base_sha256'],sampling='25% target strata intersected with exact shared text boundaries',
        weighting='equal total example mass',objective='layer relative MSE + normalized fused-context relative MSE')
    args.out.mkdir(parents=True,exist_ok=True);resume=args.out/'resume.pt'
    if resume.exists():
        saved=torch.load(resume,weights_only=True,map_location='cpu')
        assert saved['contract']==contract
        mapper.load_state_dict(saved['mapper']);optimizer.load_state_dict(saved['optimizer'])
        history=saved['history'];step=saved['step'];first_epoch=saved['epoch']+1
    for epoch in range(first_epoch,3):
        began=time.perf_counter();loss_sum=0.;seen=0
        for x,y,w in batches(items,2048,42+epoch):
            rate=(step+1)/warm if step<warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,steps-warm)))
            for group in optimizer.param_groups:group['lr']=.001*rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                weighted=(mapper.feature_loss(x.cuda(),y.cuda())*w.cuda()).sum()
                loss=weighted*(total/4096)/2048
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(mapper.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            loss_sum+=float(weighted.detach());step+=1;seen+=len(w)
        assert seen==total and step==(epoch+1)*steps_epoch
        history.append(dict(epoch=epoch+1,loss=loss_sum/4096,seconds=time.perf_counter()-began,positions=total))
        torch.save(dict(contract=contract,mapper=mapper.state_dict(),optimizer=optimizer.state_dict(),
                        epoch=epoch,step=step,history=history),resume.with_suffix('.tmp'))
        resume.with_suffix('.tmp').replace(resume)
        print(json.dumps(history[-1]),flush=True)
    state=load_file(str(base/'model.safetensors'))
    assert all(torch.equal(v,state[k]) for k,v in weights.items() if k!='fc.weight')
    state['fc.weight']=mapper.folded().detach().to(device='cpu',dtype=torch.bfloat16).contiguous()
    export=args.out/'epoch-3/export';export.mkdir(parents=True,exist_ok=True)
    save_file(state,str(export/'model.safetensors'))
    config=json.loads((base/'config.json').read_text());config['dflash_config']['target_layer_ids']=[1,8,15,22,29];config['num_target_layers']=32
    write(export/'config.json',config)
    write(args.out/'transfer.json',dict(status='complete',family='cross',target_adapters=None,input_width=20480,
        draft=str(native),base_export=str(export),base_sha256=digest(export/'model.safetensors'),
        base_training_epochs=3,records=4096,initializer_records=4096,map_checkpoint=str(resume),
        full_data_index_sha256=digest(args.index),full_data_index_path=str(args.index),feature_manifests={str(args.index):digest(args.index)},scope='full cross-family ZIP initializer; no decoding claim'))
    write(args.out/'summary.json',dict(status='feature_fit_complete',contract=contract,history=history,
        steps=step,checkpoint_sha256=digest(resume),export_sha256=digest(export/'model.safetensors'),
        frozen_non_fc_exact=True,training_seconds=sum(h['seconds'] for h in history)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['index','reference','out']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
