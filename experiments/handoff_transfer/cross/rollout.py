"""Frozen Llama8 target rollout pilot for heterogeneous-vocabulary AUF."""
import argparse
import json
import os
import time
from pathlib import Path
from experiments.auf_vllm.pilot_data import rows,write,sha


ROOT=Path("/scratch/aryama.murthy/relayspec-auf-20260911")
MODELS=ROOT/"models"


def generate(out,count=32,offset=0,dev_count=8,family="llama",validation=False):
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    target_name="llama8-source" if family == "llama" else "qwen14-target"
    tokenizer=AutoTokenizer.from_pretrained(MODELS/target_name,local_files_only=True)
    parts={"train":rows(ROOT/"manifests/train.jsonl")[offset:offset+count],"dev":rows(ROOT/"manifests/dev.jsonl")[:dev_count]}
    if validation:
        parts={"train":[],"dev":rows(ROOT/"manifests/eval.jsonl")[128+offset:128+offset+count]}
    assert len(parts["dev" if validation else "train"]) == count
    assert not ({x["group_id"] for x in parts["train"]} & {x["group_id"] for x in parts["dev"]})
    llm=LLM(model=str(MODELS/target_name),dtype="bfloat16",max_model_len=5120,
            max_num_seqs=64 if count>32 and family == "llama" else 32,max_num_batched_tokens=8192,
            gpu_memory_utilization=.8 if family == "llama" else .9,
            enable_prefix_caching=False,generation_config="vllm",
            compilation_config={"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"})
    params=SamplingParams(temperature=0,max_tokens=4096,seed=42,stop_token_ids=[tokenizer.eos_token_id])
    for split,items in parts.items():
        dest=out/f"{split}.json"
        if not items:
            write(dest,[])
            continue
        if dest.exists():
            assert [x["group_id"] for x in json.loads(dest.read_text())] == [x["group_id"] for x in items]
            continue
        prompts=[]
        for row in items:
            content=row["problem"]+"\nSolve the problem and put your final answer within \\boxed{}."
            extra={} if family == "llama" else {"enable_thinking":False}
            text=tokenizer.apply_chat_template([{"role":"user","content":content}],tokenize=False,add_generation_prompt=True,**extra)
            ids=tokenizer.encode(text,add_special_tokens=False)
            assert len(ids) <= 1024
            prompts.append(ids)
        start=time.perf_counter()
        outputs=llm.generate([{"prompt_token_ids":ids} for ids in prompts],params,use_tqdm=False)
        generated=[]
        for row,ids,response in zip(items,prompts,outputs):
            tokens=list(response.outputs[0].token_ids)
            generated.append({"row_id":row["row_id"],"group_id":row["group_id"],"prompt_token_ids":ids,
                "output_ids":tokens,"full_ids":ids+tokens,"temperature":0,
                "finish_reason":response.outputs[0].finish_reason})
        write(dest,generated)
        write(out/f"{split}-generation.json",{"seconds":time.perf_counter()-start,"sha256":sha(dest),
            "teacher":"unsloth/Llama-3.1-8B-Instruct" if family == "llama" else "Qwen/Qwen3-14B",
            "revision":"4699cc75b550f9c6f3173fb80f4703b62d946aa5" if family == "llama" else "40c069824f4251a91eefaf281ebe4c544efd3e18",
            "target_adapters":None,"records":len(items),"offset":offset,"output_cap":4096,"temperature":0,"seed":42,"source_manifest_sha256":sha(ROOT/"manifests/train.jsonl"),"job_id":os.environ.get("SLURM_JOB_ID")})
        print(json.dumps({"split":split,"generated":len(items),"seconds":time.perf_counter()-start}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int,default=16);p.add_argument('--offset',type=int,default=0);a=p.parse_args()
    assert a.count>0 and a.offset>=0
    a.out.mkdir(parents=True,exist_ok=True)
    generate(a.out,count=a.count,offset=a.offset,dev_count=0,family='llama')
