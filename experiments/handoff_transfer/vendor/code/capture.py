"""Dense NanoCoder-target features on its own saved continuations, canonical vLLM layers."""
from common import *
import argparse,os,time
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
from sampling import key
paths=sorted((R/'rollouts'/a.tag).glob('*.json'));assert paths
index={}
for path in paths:
 for x in json.loads(path.read_text())['rows']:index[key(x['full_ids'])]={'boundary':len(x['prompt_token_ids']),'group_id':x['group_id']}
ip=R/'setup'/f'capture_index_{a.tag}_{a.worker}.json';put(ip,index)
os.environ['SD_CAPTURE_INDEX']=str(ip);os.environ['SD_CAPTURE_DENSE_CHECK']='1'
import capture_runtime
capture_runtime.install()
import torch
from vllm import LLM,PoolingParams
from vllm.lora.request import LoRARequest
from runtime import llm,request
engine=llm(pool=True)
for i,path in enumerate(paths):
 if i%a.workers!=a.worker:continue
 dst=R/'features'/a.tag/(path.stem+'.pt')
 if dst.exists():continue
 rows=json.loads(path.read_text())['rows'];features=[];t=time.perf_counter()
 for start in range(0,len(rows),4):
  batch=rows[start:start+4]
  outs=engine.encode([{'prompt_token_ids':x['full_ids']} for x in batch],pooling_task='token_embed',pooling_params=PoolingParams(task='token_embed'),lora_request=request(),use_tqdm=False)
  for x,o in zip(batch,outs):
   h=o.outputs.data.to(dtype=torch.bfloat16,device='cpu');assert h.shape==(len(x['full_ids']),12800) and torch.isfinite(h).all();features.append(h.contiguous())
 dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_suffix('.tmp');torch.save({'features':features,'rows':rows,'rollout_sha256':sha(path)},tmp);tmp.replace(dst)
 put(dst.with_suffix('.json'),{'seconds':time.perf_counter()-t,'sha256':sha(dst),'bytes':dst.stat().st_size,'source_sha256':sha(path),'target':DOMAIN+'_lora'})
 print('CAPTURE',a.tag,path.stem,flush=True)
