"""Bounded cross-family integration fit; explicitly separate from full study."""
import argparse,contextlib,json,os,time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import build,export,TRANSFER,R,OBJECTIVE
from experiments.handoff_transfer.cross.online import cross_online_class
from experiments.handoff_transfer.cross.batch import collate
from experiments.handoff_transfer.matrix.prepare import digest
from experiments.auf_vllm.pilot_data import write

def main(a):
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==2
    torch.cuda.set_device(rank);dist.init_process_group('nccl');torch.manual_seed(42)
    raw,cfg=build(a.kind);raw.__class__=cross_online_class(type(raw));raw.train()
    model=DDP(raw,device_ids=[rank],broadcast_buffers=False)
    params=[p for p in raw.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=0,fused=True)
    rows=json.loads((R/'train.json').read_text());alignment=json.loads((R/'source-label-alignment.json').read_text())
    assert len(rows)==len(alignment)==TRANSFER['records']
    conditioning=json.loads((R/'conditioning-contract.json').read_text())
    assert conditioning['train_sha256']==digest(R/'train.json') and conditioning['causality_check']['passed']
    aligned={r['group_id']:r for r in alignment};cache=[]
    for i,row in enumerate(rows):
        path=R/f'features/source/train/{i:05d}.pt'
        provenance=json.loads(path.with_suffix('.json').read_text());assert provenance['sha256']==digest(path)
        data=torch.load(path,weights_only=True,mmap=True)
        assert data['group_id']==row['group_id'] and data['layers']==[1,8,15,22,29]
        cache.append({'target_ids':row['full_ids'],'features':data['features'],'alignment':aligned[row['group_id']]})
    assert len(cache)>=8 and a.steps>0
    began=time.perf_counter();history=[]
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True);losses=[]
        # Reproducible record permutations, global batch8, two microbatches.
        epoch=(step*8)//len(cache);offset=(step*8)%len(cache)
        order=torch.randperm(len(cache),generator=torch.Generator().manual_seed(42+epoch)).tolist()
        for micro in range(2):
            indices=[order[(offset+micro*4+rank*2+j)%len(cache)] for j in range(2)]
            batch=tuple(x.cuda() for x in collate([cache[i] for i in indices],raw.block_size))
            torch.manual_seed(42000+step*100+rank*10+micro)
            with model.no_sync() if micro==0 else contextlib.nullcontext():
                with torch.autocast('cuda',dtype=torch.bfloat16):loss,_,metrics=model(*batch)
                assert torch.isfinite(loss);(loss/2).backward()
            losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);opt.step()
        history.append({'step':step+1,'rank_local_mean_loss':sum(losses)/2})
    dist.barrier()
    if rank==0:
        out=R/'cross-fits'/f'{a.kind}-{OBJECTIVE}'
        export(raw,cfg,a.kind,out/'export')
        torch.save({k:p.detach().cpu() for k,p in raw.draft_model.named_parameters() if p.requires_grad},out/'parameters.pt')
        write(out/'summary.json',{'scope':'integration pilot, not full-data benchmark','records':len(cache),
            'steps':a.steps,'presentations':8*a.steps,'anchors_limit':512,'objective':OBJECTIVE,
            'label_units':'source tokens','context_units':'target tokens','lr':a.lr,
            'seconds':time.perf_counter()-began,'peak_cuda_bytes_rank0':torch.cuda.max_memory_allocated(),
            'history':history,'initializer_records':TRANSFER['initializer_records'],
            'train_sha256':digest(R/'train.json'),'alignment_sha256':digest(R/'source-label-alignment.json')})
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['fusion_r56','five_maps','five_ba56','dense_fusion'],required=True)
    p.add_argument('--steps',type=int,default=2);p.add_argument('--lr',type=float,default=1e-4);main(p.parse_args())
