import json,time,math,os
from pathlib import Path
import torch
from safetensors.torch import load_file
from paths import WORK,model
from mapper import Context
from data import CACHE,pairs,batches,rows
from sampling import positions
SPEC={'interface':'C','length':4096,'size':16384,'order':'linear','fusion':'added'}
def atomic(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.part');torch.save(obj,tmp);tmp.replace(path)
def make():
 d=load_file(model(4,'draft')/'model.safetensors');return Context(d['fc.weight'],d['hidden_norm.weight'])
def fit():
 batch=2048;epochs=3;train_count=16384
 assert torch.cuda.is_available() and json.loads((CACHE/'manifest.json').read_text())['complete']
 torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 out=WORK/'fit';out.mkdir(parents=True,exist_ok=True)
 if (out/'summary.json').exists():return
 m=make().cuda();opt=torch.optim.AdamW(m.parameters(),lr=1e-3,betas=(.9,.999),weight_decay=0,fused=True)
 def items(split):return pairs(split,train_count,4096)
 def losses(x,y):
  with torch.autocast('cuda',dtype=torch.bfloat16):return m.loss(x,y)
 total=0
 for x in rows('train',train_count):
  p=positions(len(x['full_ids']),len(x['prompt_token_ids']),x['group_id'])
  total+=sum(t<len(x['prompt_token_ids'])+4096 for t in p)
 steps_epoch=math.ceil(total/batch);steps=steps_epoch*epochs;warm=max(1,int(.05*steps))
 start=0;step=0;history=[]
 resume=out/'resume.pt'
 if resume.exists():
  q=torch.load(resume,weights_only=False);assert q['spec']==SPEC;m.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);start=q['epoch']+1;step=q['step'];history=q['history'];assert q['planned_epochs']==epochs
 wall=time.perf_counter()
 for epoch in range(start,epochs):
  m.train();epoch_wall=time.perf_counter();train_sum=0
  for x,y,w in batches(items('train'),batch,42+epoch):
   rate=(step+1)/warm if step<warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,steps-warm)))
   for g in opt.param_groups:g['lr']=1e-3*rate
   opt.zero_grad(set_to_none=True)
   # Fixed global scaling preserves equal total weight per example across batches.
   scale=(total/train_count)/batch
   micro=batch
   for k in range(0,len(x),micro):
    xx=x[k:k+micro].pin_memory().cuda(non_blocking=True);yy=y[k:k+micro].pin_memory().cuda(non_blocking=True);ww=w[k:k+micro].cuda()
    l=losses(xx,yy);assert torch.isfinite(l).all();loss=(l*ww).sum()*scale;loss.backward();train_sum+=float((l.detach()*ww).sum())
   torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step();step+=1
  event={'epoch':epoch,'train_loss':train_sum/train_count,'wall_seconds':time.perf_counter()-epoch_wall,'step':step};history.append(event);print(json.dumps(event),flush=True)
  atomic(resume,{'spec':SPEC,'model':m.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'step':step,'history':history,'planned_epochs':epochs})
 assert len(history)==epochs and history[-1]['epoch']==epochs-1
 atomic(out/'final.pt',{'spec':SPEC,'model':m.state_dict(),'epoch':epochs-1,'planned_epochs':epochs,'selection':'final_fixed_epoch'})
 (out/'summary.json').write_text(json.dumps({'spec':SPEC,'actual_train_examples':train_count,'actual_dev_examples':0,'selected_epoch':epochs-1,'selection':'final_fixed_epoch','positions_per_epoch':total,'parameters':sum(x.numel() for x in m.parameters()),'history':history,'wall_seconds_this_invocation':time.perf_counter()-wall,'job_id':os.environ.get('SLURM_JOB_ID'),'weighting':'equal example'},indent=2))
if __name__=='__main__':fit()
