import os,json,time,sys,hashlib
from pathlib import Path
from paths import WORK,PACKAGE,model
R=WORK
def main():
 from vllm import LLM,SamplingParams
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument("--count",type=int,required=True);ap.add_argument("--split",default="train");ap.add_argument("--rank",type=int,default=0);ap.add_argument("--world-size",type=int,default=1);ap.add_argument("--probe",action="store_true");ap.add_argument("--cap",type=int,default=4096);a=ap.parse_args();assert a.split=="train" and a.world_size==2 and a.cap==4096
 if not a.probe:assert json.loads((WORK/"validation/generation_gate.json").read_text())["passed"]
 P=WORK
 rows=[json.loads(x) for x in (P/"common/dataset"/(a.split+".jsonl")).read_text().splitlines()][:a.count]
 dst=(R/"validation/probe_rollouts_r1"/a.split) if a.probe else (R/"common/rollouts"/a.split);dst.mkdir(parents=True,exist_ok=True)
 missing=[i for i in range(0,len(rows),128) if (i//128)%a.world_size==a.rank and not (dst/(str(i//128).zfill(5)+".jsonl")).exists()]
 if not missing:return
 t=time.perf_counter()
 llm=LLM(model=str(model(8)),dtype="bfloat16",
 max_model_len=5120,max_num_seqs=128,max_num_batched_tokens=8192,gpu_memory_utilization=.85,
 enable_prefix_caching=False,seed=42,attention_config={"backend":"FLASH_ATTN"},disable_log_stats=False)
 setup=time.perf_counter()-t;params=SamplingParams(temperature=0,max_tokens=a.cap)
 # Warm on non-training examples, outside the measured generation shards.
 warm=[json.loads(x) for x in (P/"common/dataset/dev.jsonl").read_text().splitlines()[-4:]]
 llm.generate([{"prompt_token_ids":x["prompt_token_ids"]} for x in warm],SamplingParams(temperature=0,max_tokens=32),use_tqdm=False)
 # Keep a deeper request queue behind128 active sequences to amortize long tails.
 # Output persistence remains128 examples/shard; never regenerate completed shards.
 for group_start in range(0,len(missing),8):
  starts=missing[group_start:group_start+8]
  batch=[row for start in starts for row in rows[start:start+128]]
  t=time.perf_counter()
  outs=llm.generate([{"prompt_token_ids":x["prompt_token_ids"]} for x in batch],params,use_tqdm=False)
  elapsed=time.perf_counter()-t
  assert len(outs)==len(batch)
  offset=0
  for start in starts:
   shard_rows=rows[start:start+128];shard_outs=outs[offset:offset+len(shard_rows)];offset+=len(shard_rows)
   path=dst/(str(start//128).zfill(5)+".jsonl");tmp=path.with_suffix(".part")
   with tmp.open("w") as f:
    for row,out in zip(shard_rows,shard_outs):
     assert list(out.prompt_token_ids)==row["prompt_token_ids"]
     ids=list(out.outputs[0].token_ids)
     assert 0<len(ids)<=a.cap and len(row["prompt_token_ids"])+len(ids)<=5120
     record=dict(row_id=row["row_id"],group_id=row["group_id"],prompt_token_ids=row["prompt_token_ids"],output_ids=ids,full_ids=row["prompt_token_ids"]+ids,finish_reason=out.outputs[0].finish_reason,thinking=False,temperature=0)
     f.write(json.dumps(record)+"\n")
   tmp.replace(path)
   meta=dict(count=len(shard_rows),output_tokens=sum(len(o.outputs[0].token_ids) for o in shard_outs),prompt_tokens=sum(len(x["prompt_token_ids"]) for x in shard_rows),generation_wall_seconds=elapsed if start==starts[0] else 0,wall_attribution="queue window total recorded on first shard only",window_starts=starts,model_setup_wall_seconds=setup,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),slurm_job_id=os.environ.get("SLURM_JOB_ID"),engine="vllm 0.28.0+cu129",active_sequence_limit=128,queue_window_limit=1024,max_new_tokens=a.cap,max_model_len=5120)
   path.with_suffix(".meta.json").write_text(json.dumps(meta,indent=2))
  print("ROLLOUT_WINDOW",a.split,starts,"examples",len(batch),"seconds",elapsed,"tok/s",sum(len(o.outputs[0].token_ids) for o in outs)/elapsed,flush=True)
if __name__=="__main__":main()
