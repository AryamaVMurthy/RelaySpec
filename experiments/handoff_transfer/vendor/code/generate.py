"""Continuous vLLM batching across each worker's pending prompts; persist32-row shards."""
from runtime import *
import os,argparse,time
os.environ['SD_ROLLOUT']='1'
from vllm import SamplingParams
import benchmark_safe_r3 as b
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--count',type=int,default=4096);p.add_argument('--cap',type=int,default=4096);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
data=json.loads((R/'setup/train.json').read_text())[:a.count];directory=R/'rollouts'/a.tag;directory.mkdir(parents=True,exist_ok=True);blocks=[];pending=[]
for start in range(0,len(data),32):
 if start//32%a.workers!=a.worker or (directory/f'{start:05}.json').exists():continue
 blocks.append((start,len(pending),len(data[start:start+32])));pending.extend(data[start:start+32])
if not pending:raise SystemExit(0)
engine=llm();inputs=b.inputs(engine,pending);started=time.perf_counter();outputs=engine.generate(inputs,SamplingParams(temperature=0,top_p=1,top_k=-1,max_tokens=a.cap,seed=0),lora_request=request(),use_tqdm=True);seconds=time.perf_counter()-started
for bi,(start,offset,count) in enumerate(blocks):
 rows=[]
 for x,ins,o in zip(pending[offset:offset+count],inputs[offset:offset+count],outputs[offset:offset+count]):
  out=o.outputs[0];ids=list(out.token_ids);rows.append({'group_id':x['group_id'],'prompt_token_ids':ins['prompt_token_ids'],'output_ids':ids,'full_ids':ins['prompt_token_ids']+ids,'text':out.text,'finish_reason':out.finish_reason})
 put(directory/f'{start:05}.json',{'rows':rows,'seconds':seconds if bi==0 else 0,'timing_scope':'Total bulk-worker generation seconds stored only in first newly generated shard; not per-shard latency'});print('ROLLOUT',start,count,flush=True)
put(R/f'validation/generation_{a.tag}_{a.worker}.json',{'count':len(pending),'response_tokens':sum(len(o.outputs[0].token_ids) for o in outputs),'seconds':seconds,'continuous_batching':True,'max_active_sequences':32,'worker':a.worker})
