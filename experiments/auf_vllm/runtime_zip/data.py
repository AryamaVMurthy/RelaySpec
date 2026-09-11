"""Bounded paired-shard reading, prefix crops and equal-example weighting."""
import json,hashlib
from pathlib import Path
import torch
from paths import WORK,model
CACHE=WORK/'cache'
ROLLOUTS=WORK/'common/rollouts'

def rows(split,size=16384):
 count=0
 for p in sorted((ROLLOUTS/split).glob('*.jsonl')):
  for line in p.read_text().splitlines():
   if count>=size:return
   count+=1;yield json.loads(line)
 assert count==size

def pairs(split,size,length):
 """At most one 128-example shard in host memory; sampled positions only."""
 count=0
 for p in sorted((CACHE/'features'/split/'8').glob('*.pt')):
  if count>=size:break
  a=torch.load(p,weights_only=True);b=torch.load(CACHE/'features'/split/'4'/p.name,weights_only=True)
  assert a['group_ids']==b['group_ids'] and a['rollout_sha256']==b['rollout_sha256']
  for i,(x,y,pos,p4) in enumerate(zip(a['features'],b['features'],a['positions'],b['positions'])):
   if count>=size:break
   assert torch.equal(pos,p4);keep=pos<a['prompt_lengths'][i]+length
   assert keep.any()
   yield count,x[keep],y[keep],pos[keep],a['prompt_lengths'][i]
   count+=1
 assert count==size,(count,size)

def batches(items,batch,seed):
 """One-example mixing buffer bounded by batch plus one sequence; each example mass=1."""
 g=torch.Generator().manual_seed(seed);xs=[];ys=[];ws=[];n=0
 for _,x,y,*_ in items:
  ix=torch.randperm(len(x),generator=g);x=x[ix];y=y[ix];w=torch.full((len(x),),1/len(x))
  xs.append(x);ys.append(y);ws.append(w);n+=len(x)
  if n>=batch:
   x=torch.cat(xs);y=torch.cat(ys);w=torch.cat(ws)
   while len(x)>=batch:
    yield x[:batch],y[:batch],w[:batch]
    x=x[batch:];y=y[batch:];w=w[batch:]
   xs=[x];ys=[y];ws=[w];n=len(x)
 if n:yield torch.cat(xs),torch.cat(ys),torch.cat(ws)

def tensor(root,size,key):
 from safetensors import safe_open
 p=model(size)
 index=p/'model.safetensors.index.json'
 paths=[p/json.loads(index.read_text())['weight_map'][key]] if index.exists() else list(p.glob('*.safetensors'))
 for path in paths:
  with safe_open(path,framework='pt') as f:
   if key in f.keys():return f.get_tensor(key)
 raise KeyError(key)
