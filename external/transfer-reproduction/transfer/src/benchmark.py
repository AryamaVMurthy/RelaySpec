import os,json,time,sys,dataclasses
from pathlib import Path
from vllm import LLM,SamplingParams
from paths import WORK,model
R=WORK
DATA=WORK/'evaluation'
def serial(x):
 if dataclasses.is_dataclass(x):return dataclasses.asdict(x)
 if hasattr(x,"__dict__"):return vars(x)
 return str(x)
def counters(llm):
 return {serial(x)["name"]:serial(x).get("value",0) for x in llm.get_metrics()}
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument("mode",help="ar8, native8, or mapped");ap.add_argument("--split",default="eval")
 ap.add_argument("--count",type=int,default=128);ap.add_argument("--repeat",type=int,default=0)
 ap.add_argument("--tag",default="final");ap.add_argument("--cap",type=int,default=2048)
 ap.add_argument("offset",nargs="?",type=int,default=0);ap.add_argument("--worker-index",type=int,default=0)
 ap.add_argument("--workers",type=int,default=1)
 args=ap.parse_args();assert 0<=args.worker_index<args.workers;label=args.mode;mode=label
 assert label in ('ar8','native8','mapped')
 variant=label=='mapped'
 if variant:
  os.environ['TRANSFER_MAPPED']='1'
  import mapper_runtime
  mapper_runtime.install()
 outdir=R/"measurements"/args.tag/("worker_"+str(args.worker_index));outdir.mkdir(parents=True,exist_ok=True)
 outpath=outdir/(label+"-r"+str(args.repeat)+".jsonl")
 if outpath.exists():
  if outpath.with_suffix(".summary.json").exists():return
  quarantine=outdir/"quarantine";quarantine.mkdir(exist_ok=True)
  outpath.rename(quarantine/(outpath.name+"."+str(time.time_ns())))
 rows=[json.loads(x) for x in (DATA/(args.split+".jsonl")).read_text().splitlines()][args.offset:args.offset+args.count][args.worker_index::args.workers]
 warm=[json.loads(x) for x in (DATA/"warmup.jsonl").read_text().splitlines()][-4:]
 size='8'
 pair=model(8).parent
 assert os.environ.get("VLLM_BATCH_INVARIANT")=="1"
 kw=dict(async_scheduling=False,worker_extension_cls="metrics.MetricsWorker",model=str(pair/"target"),dtype="bfloat16",max_model_len=4096,max_num_seqs=8,
  max_num_batched_tokens=2048,gpu_memory_utilization=.8,enable_prefix_caching=False,
  seed=0,generation_config="vllm",attention_config={"backend":"FLASH_ATTN"},
  enforce_eager=os.environ.get("TRANSFER_EAGER")=="1",compilation_config={"mode":0,"cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2,4,8,16,32,64]},disable_log_stats=False)
 if mode!="ar8":
  draft=pair/"draft"
  if variant:draft=WORK/"export"
  kw["speculative_config"]={"method":"dflash","model":str(draft),"num_speculative_tokens":15}
 t=time.perf_counter();llm=LLM(**kw);setup=time.perf_counter()-t
 params=SamplingParams(temperature=0,top_p=1.0,top_k=-1,min_p=0.0,repetition_penalty=1.0,presence_penalty=0.0,frequency_penalty=0.0,seed=0,stop_token_ids=[151645],ignore_eos=False,max_tokens=args.cap)
 llm.collective_rpc("sd_install_monitor")
 attachment_check=llm.collective_rpc("sd_verify_attachments",args=(str(draft) if variant else None,int(size)))
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(pair/"target")
 refs=[]
 for question,expected in [("What is 2 + 2? Answer with only the number.","4"),("Repeat exactly: hello","hello")]:
  text=tok.apply_chat_template([{"role":"user","content":question}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  ids=tok.encode(text,add_special_tokens=False)
  o=llm.generate([{"prompt_token_ids":ids}],SamplingParams(temperature=0,top_p=1,top_k=-1,max_tokens=32,seed=0,stop_token_ids=[151645]),use_tqdm=False)[0].outputs[0]
  refs.append(dict(prompt=question,expected=expected,text=o.text,output_ids=list(o.token_ids),passed=o.text.strip()==expected))
 (outdir/(label+"-references.json")).write_text(json.dumps(refs,indent=2))
 assert all(x["passed"] for x in refs),refs

 t=time.perf_counter()
 for x in warm:
  llm.generate([{"prompt_token_ids":x["prompt_token_ids"]}],
   SamplingParams(temperature=0,top_p=1.0,top_k=-1,min_p=0.0,repetition_penalty=1.0,presence_penalty=0.0,frequency_penalty=0.0,seed=0,stop_token_ids=[151645],ignore_eos=False,max_tokens=args.cap),use_tqdm=False)
 warmtime=time.perf_counter()-t
 before=llm.collective_rpc("sd_stats")
 metrics_before=llm.get_metrics()
 totalstart=time.perf_counter()
 with outpath.open("x") as f:
  for i,x in enumerate(rows):
   status0=llm.collective_rpc("sd_stats")[0]
   counts0=counters(llm)
   start=time.perf_counter()
   o=llm.generate([{"prompt_token_ids":x["prompt_token_ids"]}],params,use_tqdm=False)[0]
   wall=time.perf_counter()-start
   status1=llm.collective_rpc("sd_stats")[0]
   cold_wall=None
   if any(status0[k]!=status1[k] for k in ("jit_events","teacher_graph_captures")):
    cold_wall=wall
    counts0=counters(llm)
    start=time.perf_counter()
    o=llm.generate([{"prompt_token_ids":x["prompt_token_ids"]}],params,use_tqdm=False)[0]
    wall=time.perf_counter()-start
    status2=llm.collective_rpc("sd_stats")[0]
    assert all(status1[k]==status2[k] for k in ("jit_events","teacher_graph_captures")),"Repeated runtime compilation"
   counts1=counters(llm)
   delta=lambda k:counts1.get("vllm:"+k,0)-counts0.get("vllm:"+k,0)
   drafts=delta("spec_decode_num_drafts");accepted=delta("spec_decode_num_accepted_tokens")
   row=dict(verification_iterations=drafts,accepted_draft_tokens=accepted,accepted_plus_bonus_proxy=(1+accepted/drafts) if drafts else None,cold_wall_seconds=cold_wall,timing_valid=True,index=args.offset+args.workers*i+args.worker_index,worker_index=args.worker_index,row_id=x["row_id"],group_id=x["group_id"],mode=label,
    repeat=args.repeat,wall_seconds=wall,prompt_tokens=len(x["prompt_token_ids"]),
    output_tokens=len(o.outputs[0].token_ids),output_ids=list(o.outputs[0].token_ids),prompt_ids=x["prompt_token_ids"],output_text=o.outputs[0].text,
    finish_reason=o.outputs[0].finish_reason,metrics=serial(o.metrics) if o.metrics else None)
   f.write(json.dumps(row,default=serial)+"\n");f.flush()
   if (i+1)%16==0:print("PROGRESS",mode,i+1,"elapsed",time.perf_counter()-totalstart,flush=True)
 after=llm.collective_rpc("sd_stats")
 summary=dict(mode=label,config=kw,args=vars(args),setup_wall_seconds=setup,
  gpu_identity=llm.collective_rpc("sd_stats"),attachment_check=attachment_check,
  warmup_wall_seconds=warmtime,measurement_wall_seconds=time.perf_counter()-totalstart,
  gpu_before=before,gpu_after=after,metrics_before=metrics_before,metrics_after=llm.get_metrics(),
  slurm_job_id=os.environ.get("SLURM_JOB_ID"),cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
  timing_valid=True,timing_contract="Per-request generate wall; compile/capture-affected request retried once, cold time retained; prefix cache off")
 (outdir/(label+"-r"+str(args.repeat)+".summary.json")).write_text(json.dumps(summary,default=serial,indent=2))
 print("BENCH_DONE",mode,summary["measurement_wall_seconds"],summary["timing_valid"],flush=True)
if __name__=="__main__":main()
