"""Two GPUs per variant; identical global example/anchor budgets and final checkpoints."""
import os,contextlib
from model import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def batches(tag,epoch,rank,world):
 paths=sorted((PREV/'features'/tag).glob('*.pt'));rng=random.Random(42+epoch);rng.shuffle(paths)
 for pi,path in enumerate(paths):
  s=torch.load(path,weights_only=True,mmap=True)
  indices=list(range(len(s['rows'])));rng.shuffle(indices);indices.sort(key=lambda i:len(s['rows'][i]['full_ids'])//64)
  for start in range(0,len(indices),2):
   # Every rank participates even for the one-shard bounded check.
   if (pi*16+start//2)%world!=rank:continue
   ii=indices[start:start+2];data=[s['rows'][i] for i in ii];S=max(len(x['full_ids']) for x in data)
   ids=torch.zeros((len(ii),S),dtype=torch.long);h=torch.zeros((len(ii),S,INPUT_WIDTH),dtype=torch.bfloat16);mask=torch.zeros((len(ii),S))
   for j,(i,x) in enumerate(zip(ii,data)):
    n=len(x['full_ids']);assert n-len(x['prompt_token_ids'])<=4096;ids[j,:n]=torch.tensor(x['full_ids']);h[j,:n]=s['features'][i];mask[j,len(x['prompt_token_ids']):n]=1
   yield ids.cuda(),h.cuda(),mask.cuda()

def main(a):
 rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==2
 torch.cuda.set_device(rank);dist.init_process_group('nccl');torch.manual_seed(a.seed)
 raw,cfg=build(a.kind);raw.train();model=DDP(raw,device_ids=[rank],broadcast_buffers=False)
 parameters=[p for p in raw.parameters() if p.requires_grad]
 anchor_count=0;records_count=0
 n=sum(p.numel() for p in parameters);groups=[{'params':[p],'lr':.01 if name=='draft_model.fc.bias' else a.lr,'base_lr':.01 if name=='draft_model.fc.bias' else a.lr} for name,p in raw.named_parameters() if p.requires_grad];opt=torch.optim.AdamW(groups,weight_decay=0,fused=True)
 count=32 if a.tag.startswith('check') else 4096;total_steps=a.max_steps or count//8*a.epochs;step=0;history=[];t0=time.perf_counter();opt.zero_grad(set_to_none=True)
 resume_path=R/'modules'/(a.output_tag or a.tag)/a.kind/'resume.pt'
 resume=None
 if resume_path.exists():
  resume=torch.load(resume_path,weights_only=False,map_location='cpu')
  raw.draft_model.load_state_dict(resume['parameters'],strict=False);opt.load_state_dict(resume['optimizer'])
  step=resume['step'];history=resume['history'];t0-=resume['elapsed_seconds']
 for epoch in range(a.epochs):
  if resume and epoch<resume['epoch']:continue
  starttime=time.perf_counter();accum=0;loss_sum=0;num=0
  for batch_index,(ids,h,mask) in enumerate(batches(a.tag,epoch,rank,world)):
   if resume and epoch==resume['epoch'] and batch_index<resume['next_batch']:
    num+=1
    loss_sum=resume['loss_sums'][rank]
    continue
   anchor_count+=int(((mask[:,:-1]>.5)&(mask[:,1:]>.5)).sum(dim=1).clamp(max=512).sum());records_count+=ids.shape[0]
   torch.manual_seed(a.seed*1000+epoch*10000+rank*3000+num)
   with model.no_sync() if accum==0 else contextlib.nullcontext():
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,acc,metrics=model(ids,h,mask,collect_detailed_metrics=False)
    assert torch.isfinite(loss);(loss/2).backward()
   num+=1;accum+=1;loss_sum+=float(loss.detach())
   if accum==2:
    grad=torch.nn.utils.clip_grad_norm_(parameters,1);assert torch.isfinite(grad)
    step+=1
    lr=a.lr*min(1,step/max(1,.05*total_steps))*.5*(1+math.cos(math.pi*max(0,(step-.05*total_steps)/(.95*total_steps))))
    for g in opt.param_groups:g['lr']=lr*g['base_lr']/a.lr
    opt.step();opt.zero_grad(set_to_none=True);accum=0
    if step % a.snapshot_every == 0:
     if rank==0:
      processed=step*8;out=R/'modules'/(a.output_tag or a.tag)/f'seen_{processed}';out.mkdir(parents=True,exist_ok=True)
      parameters_snapshot={k:p.detach().cpu().clone() for k,p in raw.draft_model.named_parameters() if p.requires_grad}
      torch.save({'parameters':parameters_snapshot,'optimizer':opt.state_dict(),'step':step,'processed_examples':processed,'seed':a.seed,'epoch_index':epoch,'next_batch':batch_index+1,'anchor_rng':'seed*1000+epoch*10000+rank*3000+microbatch_index'},out/'checkpoint.tmp')
      (out/'checkpoint.tmp').replace(out/'checkpoint.pt')
      torch.save(parameters_snapshot,out/'final.pt')
      export(raw,cfg,a.kind,R/'exports'/(a.output_tag or a.tag)/f'seen_{processed}')
      put(out/'summary.json',{'processed_examples':processed,'optimizer_steps':step,'unique_cached_examples':4096,'training_seconds':time.perf_counter()-t0,'training_gpu_hours':(time.perf_counter()-t0)*world/3600,'checkpoint':'fixed cumulative-example milestone','objective':'AUF','anchors_per_example':512,'seed':a.seed,'lr':a.lr,'total_schedule_steps':total_steps})
     dist.barrier()

    if step%100==0:
     loss_totals=[None]*world;dist.all_gather_object(loss_totals,loss_sum)
     if rank==0:
      resume_path.parent.mkdir(parents=True,exist_ok=True)
      state={'parameters':{k:p.detach().cpu() for k,p in raw.draft_model.named_parameters() if p.requires_grad},'optimizer':opt.state_dict(),'step':step,'epoch':epoch,'next_batch':batch_index+1,'loss_sums':loss_totals,'history':history,'elapsed_seconds':time.perf_counter()-t0,'anchor_rng':'manual seed derived from seed/epoch/rank/microbatch index before each forward'}
      torch.save(state,resume_path.with_suffix('.tmp'));resume_path.with_suffix('.tmp').replace(resume_path)
     dist.barrier()
    if rank==0 and (step%10==0 or step==1):
     status={'epoch':epoch+1,'step':step,'total_steps':total_steps,'loss_local':float(loss),'anchor_count_rank0':anchor_count,'records_rank0':records_count,'peak_cuda_bytes_rank0':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.perf_counter()-t0}
     put(R/'modules'/(a.output_tag or a.tag)/a.kind/'progress.json',status);print('TRAIN',a.kind,json.dumps(status),flush=True)
   if a.max_steps and step>=a.max_steps:break
  assert accum==0
  sums=torch.tensor([loss_sum,num],device='cuda');dist.all_reduce(sums)
  history.append({'epoch':epoch+1,'mean_batch_loss':float(sums[0]/sums[1]),'microbatches':int(sums[1]),'seconds':time.perf_counter()-starttime})
  if a.max_steps and step>=a.max_steps:break
 dist.barrier();seconds=time.perf_counter()-t0
 if rank==0:
  out=R/'modules'/(a.output_tag or a.tag)/a.kind;out.mkdir(parents=True,exist_ok=True)
  torch.save({k:p.detach().cpu() for k,p in raw.draft_model.named_parameters() if p.requires_grad},out/'final.pt')
  export(raw,cfg,a.kind,R/'exports'/(a.output_tag or a.tag)/a.kind)
  put(out/'summary.json',{'trainable_parameters':n,'optimizer_steps':step,'training_seconds':seconds,'training_gpu_hours':seconds*world/3600,'world_size':world,'seed':a.seed,'variant':a.kind,'history':history,'peak_cuda_bytes_rank0':torch.cuda.max_memory_allocated(),'lr':a.lr,'bias_lr':.01,'initialization':'ZIP epoch-3 initialization; see architecture model contract; seed42','processed_examples':step*8,'total_schedule_steps':total_steps,'objective':'accept-until-fail, first failure included, detached prefix weights','anchors_per_example':512,'batch_per_gpu':2,'gradient_accumulation':2,'checkpoint':'final','actual_anchors_rank0':anchor_count,'records_rank0':records_count,'anchor_limit_semantics':'distinct valid anchors; short records supply fewer','source_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}})
  print('TRAIN_DONE',a.kind,seconds,flush=True)
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--snapshot-every',type=int,default=500);p.add_argument('--kind',required=True,choices=VARIANTS);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--tag',default='full');p.add_argument('--output-tag',default=None);p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=3);p.add_argument('--max-steps',type=int,default=0);main(p.parse_args())
