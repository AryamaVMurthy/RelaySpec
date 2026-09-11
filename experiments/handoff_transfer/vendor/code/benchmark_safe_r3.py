"""Unmodified inputs/timing functions extracted from the validated benchmark; runtime supplied by evaluate.py."""
import os,json,time,argparse,hashlib,math

from pathlib import Path

def put(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.part');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)

def inputs(engine,data):
 tok=engine.get_tokenizer()
 return [{'prompt_token_ids':tok.apply_chat_template(x['messages'],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)} for x in data]

def bench(a):
 from vllm import SamplingParams
 setup=time.perf_counter();engine=llm(spec=a.side!='ar',draft=R/'exports'/a.tag if a.side=='mapped' else None);data=rows('eval')[:a.count][a.worker::a.workers];ins=inputs(engine,data);params=SamplingParams(temperature=0,max_tokens=a.cap,seed=0)
 if a.side!='ar':engine.collective_rpc('sd_assert_sharing')
 engine.generate(inputs(engine,rows('train')[:2]),SamplingParams(temperature=0,max_tokens=32),lora_request=request(),use_tqdm=False)
 setup_seconds=time.perf_counter()-setup;engine.collective_rpc('sd_install_monitor');result=[];dst=R/'measurements'/(a.results_tag or a.tag)/(f'{a.side}.json' if a.workers==1 else f'{a.side}_part{a.worker}.json')
 import sys
 sys.path.insert(0,str(R.parent))
 from acceptance_metrics import snapshot,delta
 from resume_state import restored
 result=restored(dst,data,a);resume_rows=len(result)
 assert a.side=='ar' or all('verification_iterations' in x and 'proposed_draft_tokens' in x for x in result),'Use a fresh --results-tag; older rows have no acceptance counters'
 for x,i in zip(data[resume_rows:],ins[resume_rows:]):
  counts0=snapshot(engine) if a.side!='ar' else None
  before=engine.collective_rpc('sd_stats');t=time.perf_counter();o=engine.generate([i],params,lora_request=request(),use_tqdm=False)[0].outputs[0];dt=time.perf_counter()-t
  after=engine.collective_rpc('sd_stats');cold=None
  if before[0]['jit_events']!=after[0]['jit_events']:
   cold=dt
   counts0=snapshot(engine) if a.side!='ar' else None
   t=time.perf_counter();o=engine.generate([i],params,lora_request=request(),use_tqdm=False)[0].outputs[0];dt=time.perf_counter()-t
   assert engine.collective_rpc('sd_stats')[0]['jit_events']==after[0]['jit_events']
  acceptance=delta(counts0,snapshot(engine)) if counts0 is not None else {}
  result.append({**acceptance,'group_id':x['group_id'],'cold_seconds':cold,'device':after,'seconds':dt,'output_ids':list(o.token_ids),'text':o.text,'finish_reason':o.finish_reason,'reference':x['reference']});put(dst,{'rows':result,'complete':len(result)==len(data),'backend':'vllm','compilation_mode':0,'cudagraph_mode':'FULL_DECODE_ONLY','requested_cudagraph_mode':'FULL','batch_invariant':True,'async_scheduling':False,'cap':a.cap,'worker':a.worker,'workers':a.workers,'side':a.side,'resumed_existing_rows':resume_rows,'setup_and_warmup_seconds':setup_seconds});print('EVAL',a.side,len(result),dt,flush=True)
