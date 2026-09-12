"""Full cross-family fits using frozen full-data initializers and streamed features."""
import argparse,contextlib,json,os,time,math
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors.torch import load_file
from model import build,export,TRANSFER,R,OBJECTIVE,NUM_ANCHORS
from experiments.handoff_transfer.cross.online import cross_online_class
from experiments.handoff_transfer.cross.batch import collate
from experiments.handoff_transfer.cross.dataset import CrossRecords
from experiments.handoff_transfer.cross.order import epoch_order
from experiments.handoff_transfer.matrix.prepare import digest
from experiments.auf_vllm.pilot_data import write

def main(a):
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==2
    torch.cuda.set_device(rank);dist.init_process_group('nccl');torch.manual_seed(42)
    raw,cfg=build(a.kind);raw.__class__=cross_online_class(type(raw));raw.train()
    model=DDP(raw,device_ids=[rank],broadcast_buffers=False)
    params=[p for p in raw.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=0,fused=True)
    initial={k:p.detach().cpu().clone() for k,p in raw.draft_model.named_parameters() if p.requires_grad}
    assert TRANSFER['records']==4096 and TRANSFER['initializer_records']==4096, 'Full data and initializer required'
    assert TRANSFER['full_data_index_sha256']==digest(a.index)
    cache=CrossRecords(a.index)
    assert len(cache)==4096 and a.steps in (100,500,2000)
    out=R/'modules'/f'steps-{a.steps}'/a.kind
    export_dir=R/'exports'/f'steps-{a.steps}'/a.kind
    resume_path=out/'resume.pt'
    schedule={'kind':a.kind,'objective':OBJECTIVE,'lr':a.lr,'steps':a.steps,'seed':42,
              'index_sha256':digest(a.index),'initializer_sha256':TRANSFER['base_sha256'],'record_order':'seed42+epoch shuffled64-record chunks, shuffled records within chunk'}
    first_step=0;elapsed=0.;history=[];anchor_count=0;record_count=0
    if resume_path.exists():
        resume=torch.load(resume_path,map_location='cpu',weights_only=False)
        assert resume['schedule']==schedule
        raw.draft_model.load_state_dict(resume['parameters'],strict=False)
        opt.load_state_dict(resume['optimizer'])
        first_step=resume['step'];elapsed=resume['elapsed'];history=resume['history']
        if rank==0:anchor_count=resume['anchors_rank0'];record_count=resume['records_rank0']
        assert first_step<=a.steps
    began=time.perf_counter()-elapsed
    for step in range(first_step,a.steps):
        opt.zero_grad(set_to_none=True);losses=[]
        # Reproducible record permutations, global batch8, two microbatches.
        epoch=(step*8)//len(cache);offset=(step*8)%len(cache)
        order=epoch_order(len(cache),epoch)
        for micro in range(2):
            indices=[order[(offset+micro*4+rank*2+j)%len(cache)] for j in range(2)]
            batch=tuple(x.cuda() for x in collate([cache[i] for i in indices],raw.block_size))
            if rank==0:
                anchor_count+=int(batch[2].sum(dim=1).clamp(max=NUM_ANCHORS).sum());record_count+=len(indices)
            torch.manual_seed(42000+step*100+rank*10+micro)
            with model.no_sync() if micro==0 else contextlib.nullcontext():
                with torch.autocast('cuda',dtype=torch.bfloat16):loss,_,metrics=model(*batch)
                assert torch.isfinite(loss);(loss/2).backward()
            losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True)
        update=step+1
        rate=a.lr*min(1,update/max(1,.05*a.steps))*.5*(1+math.cos(math.pi*max(0,(update-.05*a.steps)/(.95*a.steps))))
        for group in opt.param_groups:group['lr']=rate
        opt.step()
        history.append({'step':step+1,'rank_local_mean_loss':sum(losses)/2})
        if (step+1)%100==0:
            if rank==0:
                out.mkdir(parents=True,exist_ok=True)
                torch.save({'schedule':schedule,'step':step+1,'elapsed':time.perf_counter()-began,
                    'history':history,'anchors_rank0':anchor_count,'records_rank0':record_count,'parameters':{k:p.detach().cpu() for k,p in raw.draft_model.named_parameters() if p.requires_grad},
                    'optimizer':opt.state_dict()},resume_path.with_suffix('.tmp'))
                resume_path.with_suffix('.tmp').replace(resume_path)
            dist.barrier()
    dist.barrier()
    training_seconds=time.perf_counter()-began
    if rank==0:
        assert all(torch.isfinite(p).all() for p in params)
        assert any(not torch.equal(initial[k],p.detach().cpu()) for k,p in raw.draft_model.named_parameters() if p.requires_grad)
        export(raw,cfg,a.kind,export_dir)
        exported=load_file(str(export_dir/'model.safetensors'))
        base=load_file(str(Path(TRANSFER['base_export'])/'model.safetensors'))
        assert set(exported)==set(base)
        assert all(torch.equal(v,exported[k]) for k,v in base.items() if k!='fc.weight')
        fc=raw.draft_model.fc
        folded=fc.folded() if a.kind!='fusion_r56' else fc.base_layer.weight.float()+fc.get_delta_weight('default').float()
        folded=folded.detach().to(dtype=torch.bfloat16,device='cpu')
        torch.testing.assert_close(folded,exported['fc.weight'],rtol=0,atol=0)
        write(out/'verification.json',{'status':'passed','frozen_non_fc_exact':True,
            'trainable_names':sorted(initial),'export_sha256':digest(export_dir/'model.safetensors'),
            'inference_exactness':'not yet evaluated'})
        torch.save({k:p.detach().cpu() for k,p in raw.draft_model.named_parameters() if p.requires_grad},out/'parameters.pt')
        write(out/'summary.json',{'scope':'full-data cross-family fit; decoding measurement separate','records':len(cache),
            'steps':a.steps,'optimizer_steps':a.steps,'presentations':8*a.steps,'processed_examples':8*a.steps,'anchors_limit':NUM_ANCHORS,'anchors_per_example':NUM_ANCHORS,'world_size':world,'seed':42,'variant':a.kind,'checkpoint':'final','objective':OBJECTIVE,
            'label_units':'source tokens','context_units':'target tokens','lr':a.lr,
            'training_seconds':training_seconds,'training_gpu_hours':training_seconds*world/3600,'actual_anchors_rank0':anchor_count,'records_rank0':record_count,'peak_cuda_bytes_rank0':torch.cuda.max_memory_allocated(),
            'history':history,'initializer_records':TRANSFER['initializer_records'],
            'full_data_index_sha256':digest(a.index),'schedule':schedule,'trainable_parameters':sum(p.numel() for p in params),'source_sha256':digest(Path(__file__))})
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['normal_ce','fusion_r56','five_maps','five_ba56','dense_fusion','dense_fresh'],required=True)
    p.add_argument('--index',type=Path,required=True);p.add_argument('--steps',type=int,choices=[100,500,2000],required=True);p.add_argument('--lr',type=float,default=1e-4);main(p.parse_args())
