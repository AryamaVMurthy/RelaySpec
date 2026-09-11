"""Normal RelaySpec, same records/positions/epochs as the older ZIP control."""
import argparse,json,math,os,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
from experiments.auf_vllm.pilot_data import write,sha
from experiments.auf_vllm.runtime_zip.sampling import positions
from experiments.handoff_transfer.normal_interface import NormalInterface

def timed_batches(iterator, timing):
    """Measure time spent obtaining CPU feature batches without changing order."""
    iterator = iter(iterator)
    while True:
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            timing['feature_batch_wait_seconds'] += time.perf_counter() - start
            return
        timing['feature_batch_wait_seconds'] += time.perf_counter() - start
        yield batch


def main(a):
    transfer=json.loads((a.root/'transfer.json').read_text())
    assert transfer['status']=='complete' and transfer['target_adapters'] is None
    base=Path(transfer['base_export']);native=Path(transfer['draft'])
    assert sha(base/'model.safetensors')==transfer['base_sha256']
    base_config=json.loads((base/'config.json').read_text())
    native_config=json.loads((native/'config.json').read_text())
    target_config=json.loads((a.target/'config.json').read_text())
    weights=load_file(str(native/'model.safetensors'))
    if a.family=='q8':
        sys.path.insert(0,str(Path(__file__).parents[1]/'auf_vllm/runtime_zip'))
        from data import CACHE,rows,pairs,batches
        assert json.loads((CACHE/'manifest.json').read_text())['complete']
        selected=list(rows('train',4096))
        expected=json.loads((a.study/'main-q8-n4096/train.json').read_text())
        assert [(r['group_id'],r['full_ids']) for r in selected]==[(r['group_id'],r['full_ids']) for r in expected]
        iterator=lambda seed:batches(pairs('train',4096,4096),2048,seed)
    else:
        from experiments.auf_vllm.family_training import FamilyRecords,paired_batches
        name='main-l3-n4096' if a.family=='llama' else 'main-q14-n4096'
        records=FamilyRecords(a.study/name,4096)
        selected=records.rows
        iterator=lambda seed:paired_batches(records,seed)
    assert len(selected)==4096 and len({r['group_id'] for r in selected})==4096
    total=sum(len(positions(len(r['full_ids']),len(r['prompt_token_ids']),r['group_id'])) for r in selected)
    torch.manual_seed(a.seed)
    model=NormalInterface(weights['fc.weight'],weights['hidden_norm.weight'],transfer['input_width']//5,
                          target_config['rms_norm_eps'],native_config['rms_norm_eps']).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0,fused=True)
    out=a.root/'normal';out.mkdir(parents=True,exist_ok=True)
    steps_epoch=math.ceil(total/2048);steps=steps_epoch*a.epochs;warm=max(1,int(.05*steps))
    contract={'family':a.family,'records':4096,'epochs':a.epochs,'seed':a.seed,'lr':a.lr,
      'positions':total,'position_batch':2048,'input_eps':target_config['rms_norm_eps'],
      'output_eps':native_config['rms_norm_eps'],'trainable_parameters':sum(p.numel() for p in model.parameters()),
      'objective':'normal RelaySpec normalized context relative_interface_mse; equal-record weighting',
      'initialization':'original random nn.Linear; not ZIP warm-start','base_export_sha256':transfer['base_sha256'],
      'feature_manifests':transfer['feature_manifests'],'sampler':'same deterministic stratified25% positions and equal-record masses as ZIP',
      'optimizer':'AdamW betas.9/.999 eps1e-8 weight_decay0 clip1, ZIP warmup/cosine'}
    checkpoint=out/'resume.pt';step=0;first=0;history=[]
    if checkpoint.exists():
        saved=torch.load(checkpoint,weights_only=True,map_location='cuda')
        assert saved['contract']==contract
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        step,first,history=saved['step'],saved['epoch']+1,saved['history']
    write(out/'contract.json',contract)
    for epoch in range(first,a.epochs):
        start=time.perf_counter();loss_sum=0;timing={'feature_batch_wait_seconds':0.0}
        for index,(x,y,w) in enumerate(timed_batches(iterator(a.seed+epoch),timing)):
            x,y,w=x.cuda(),y.cuda(),w.cuda()
            rate=(step+1)/warm if step<warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,steps-warm)))
            opt.param_groups[0]['lr']=a.lr*rate;opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                weighted=(model.feature_loss(x,y)*w).sum()
                loss=weighted*(total/4096)/2048
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
            step+=1;loss_sum+=weighted.item()
        assert index+1==steps_epoch
        event={'epoch':epoch+1,'step':step,'seconds':time.perf_counter()-start,'loss':loss_sum/4096}
        event.update(timing)
        event['remaining_fit_seconds']=event['seconds']-timing['feature_batch_wait_seconds']
        event['timing_scope']='CPU feature-batch wait versus remaining transfer/forward/backward/optimizer/Python time; not isolated GPU kernel time.'
        history.append(event);print(json.dumps(event),flush=True)
        temp=checkpoint.with_suffix('.tmp')
        torch.save({'contract':contract,'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'step':step,'history':history},temp);temp.replace(checkpoint)
    x=next(iter(iterator(a.seed)))[0][:256].cuda()
    dest=out/'export';dest.mkdir(parents=True,exist_ok=True)
    state=load_file(str(base/'model.safetensors'))
    state['fc.weight']=model.relay.projection.weight.detach().to(dtype=torch.bfloat16,device='cpu').contiguous()
    save_file(state,str(dest/'model.safetensors'))
    config=dict(base_config);config.update(architectures=['NormalRelayDFlash'],relayspec_normal_input_eps=contract['input_eps'])
    write(dest/'config.json',config)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        ref=model(x[:256])
        projection=F.linear(F.rms_norm(x[:256],(x.shape[-1],),eps=contract['input_eps']),state['fc.weight'].cuda())
        actual=model.normalize(projection)
        error=float(((ref.float()-actual.float()).square().sum(-1)/ref.float().square().sum(-1).clamp_min(1e-20)).mean())
        assert error<1e-3,error
    write(out/'summary.json',{'status':'complete','contract':contract,'history':history,'training_seconds':sum(r['seconds'] for r in history),
      'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'export_relative_mse':error,'export_sha256':sha(dest/'model.safetensors'),
      'job_id':os.environ.get('SLURM_JOB_ID')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['q8','q14','llama'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--study',type=Path,required=True);p.add_argument('--target',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float,default=1e-3);main(p.parse_args())
