"""vLLM capture only. Original rollouts/checkpoints remain read-only."""
import os,json,time,hashlib,argparse
from pathlib import Path
from sampling import key,positions,check
from paths import WORK,model
R=WORK/'cache'
SRC=WORK/'common/rollouts'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def prepare():
 index={};totals={};examples=[]
 for split in ['train']:
  n=tok=sample=0
  for p in sorted((SRC/split).glob('*.jsonl')):
   for line in p.read_text().splitlines():
    x=json.loads(line);ids=x['full_ids'];b=len(x['prompt_token_ids'])
    assert ids==x['prompt_token_ids']+x['output_ids']
    v={'boundary':b,'group_id':x['group_id']};k=key(ids)
    assert k not in index or index[k]==v
    index[k]=v;n+=1;tok+=len(ids);sample+=len(positions(len(ids),b,x['group_id']))
    examples.append(x)
  assert n=={'train':16384,'dev':256}[split]
  totals[split]={'examples':n,'full_tokens':tok,'sampled_positions':sample,'bf16_pair_bytes':sample*66560}
 # Cover shortest/longest, capped/EOS, partial strata, and both segment boundaries.
 selected={}
 for f in [lambda x:len(x['full_ids']),lambda x:len(x['output_ids']),lambda x:len(x['prompt_token_ids'])]:
  for x in [min(examples,key=f),max(examples,key=f)]:selected[x['group_id']]=x
 for x in examples:
  if len(selected)>=8:break
  selected[x['group_id']]=x
 write(R/'setup/index.json',index);write(R/'validation/rows.json',list(selected.values()));write(R/'setup/data_contract.json',{'source':str(SRC),'fraction':.25,'sampling':'seed42 one random position per four, separately prompt/response; fixed across epochs/models; final partial stratum included','totals':totals,'cpu_validation':check(),'dense_cache':False,'layers':[1,9,17,25,33]})
 print(json.dumps(totals),flush=True)
def capture(a):
 import torch
 from vllm import LLM,PoolingParams
 os.environ['SD_CAPTURE_INDEX']=str(R/'setup/index.json')
 dense=a.mode=='dense';os.environ['SD_CAPTURE_DENSE_CHECK']='1' if dense else '0'
 if a.mode=='full':assert json.loads((R/'validation/gate.json').read_text())['passed']
 llm=LLM(model=str(model(a.size)),enforce_eager=True,runner='pooling',hf_overrides={'architectures':['CaptureQwen3ForCausalLM']},pooler_config={'task':'token_embed'},dtype='bfloat16',max_model_len=5120,max_num_seqs=16,max_num_batched_tokens=8192,gpu_memory_utilization=.8,enable_chunked_prefill=False,enable_prefix_caching=False,attention_config={'backend':'FLASH_ATTN'})
 tasks=[('validation',None,json.loads((R/'validation/rows.json').read_text()))] if a.mode!='full' else [(split,p,None) for split in ['train'] for p in sorted((SRC/split).glob('*.jsonl'))]
 for split,p,rows in tasks:
  dst=R/('features/'+split+'/'+a.size if a.mode=='full' else 'validation/'+a.mode+'/'+a.size);dst.mkdir(parents=True,exist_ok=True)
  out=dst/((p.stem if p else 'check')+'.pt')
  if out.exists():
   m=json.loads(out.with_suffix('.meta.json').read_text());assert sha(out)==m['sha256'];continue
  if rows is None:rows=[json.loads(x) for x in p.read_text().splitlines()]
  t=time.perf_counter();zs=[];ps=[]
  # Bound host transfer and output accumulation: at most 128 examples per persisted shard.
  for start in range(0,len(rows),8):
   batch=rows[start:start+8]
   outputs=llm.encode([{'prompt_token_ids':x['full_ids']} for x in batch],pooling_task='token_embed',pooling_params=PoolingParams(task='token_embed'),use_tqdm=False)
   for x,o in zip(batch,outputs):
    ids=x['full_ids'];idx=list(range(len(ids))) if dense else positions(len(ids),len(x['prompt_token_ids']),x['group_id'])
    z=o.outputs.data.to(torch.bfloat16).cpu();assert z.shape==(len(idx),5*({'4':2560,'8':4096}[a.size]));assert torch.isfinite(z).all()
    zs.append(z);ps.append(torch.tensor(idx))
  cap=time.perf_counter()-t;t=time.perf_counter();tmp=out.with_suffix('.part')
  torch.save({'features':zs,'positions':ps,'group_ids':[x['group_id'] for x in rows],'prompt_lengths':[len(x['prompt_token_ids']) for x in rows],'full_lengths':[len(x['full_ids']) for x in rows],'rollout_sha256':sha(p) if p else key([x['full_ids'] for x in rows]),'layers':[1,9,17,25,33],'sampling':'segment-stratified-25pct-seed42' if not dense else 'dense-validation-only'},tmp);tmp.replace(out)
  m={'sha256':sha(out),'bytes':out.stat().st_size,'examples':len(rows),'positions':sum(len(x) for x in ps),'capture_seconds':cap,'write_hash_seconds':time.perf_counter()-t,'job_id':os.environ.get('SLURM_JOB_ID')};write(out.with_suffix('.meta.json'),m);print(str(out.relative_to(R)),json.dumps(m),flush=True)
def audit():
 import torch
 report=[]
 for size in ['4','8']:
  a=torch.load(R/('validation/sample/'+size+'/check.pt'),weights_only=True);b=torch.load(R/('validation/dense/'+size+'/check.pt'),weights_only=True)
  assert a['group_ids']==b['group_ids']
  for x,y,p in zip(a['features'],b['features'],a['positions']):
   y=y[p];err=((x.float()-y.float()).square().sum()/y.float().square().sum().clamp_min(1e-12)).item()
   assert err<1e-6,err;report.append({'size':size,'relative_mse':err,'positions':len(p)})
 a=torch.load(R/'validation/sample/4/check.pt',weights_only=True);b=torch.load(R/'validation/sample/8/check.pt',weights_only=True)
 assert a['group_ids']==b['group_ids'] and all(torch.equal(x,y) for x,y in zip(a['positions'],b['positions']))
 write(R/'validation/comparison.json',{'passed':True,'comparisons':report,'cpu':check()});print(json.dumps(report))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','sample','dense','full','audit']);ap.add_argument('size',nargs='?',choices=['4','8']);a=ap.parse_args()
 if a.mode=='prepare':prepare()
 elif a.mode=='audit':audit()
 else:capture(a)
