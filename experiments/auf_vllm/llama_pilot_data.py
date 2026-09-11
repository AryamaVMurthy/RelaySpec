"""Original Llama3-target rollouts and paired same-token feature capture."""
import argparse
import json
import os
import time
from pathlib import Path
from .pilot_data import rows,write,sha


ROOT=Path("/scratch/aryama.murthy/relayspec-auf-20260911")
MODELS=ROOT/"models"


def generate(out):
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    tokenizer=AutoTokenizer.from_pretrained(MODELS/"llama3-target",local_files_only=True)
    parts={"train":rows(ROOT/"manifests/train.jsonl")[:32],"dev":rows(ROOT/"manifests/dev.jsonl")[:8]}
    assert not ({x["group_id"] for x in parts["train"]} & {x["group_id"] for x in parts["dev"]})
    llm=LLM(model=str(MODELS/"llama3-target"),dtype="bfloat16",max_model_len=5120,
            max_num_seqs=32,max_num_batched_tokens=8192,gpu_memory_utilization=.8,
            enable_prefix_caching=False,generation_config="vllm",
            compilation_config={"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"})
    params=SamplingParams(temperature=0,max_tokens=4096,seed=42,stop_token_ids=[tokenizer.eos_token_id])
    for split,items in parts.items():
        dest=out/f"{split}.json"
        if dest.exists():
            assert [x["group_id"] for x in json.loads(dest.read_text())] == [x["group_id"] for x in items]
            continue
        prompts=[]
        for row in items:
            content=row["problem"]+"\nSolve the problem and put your final answer within \\boxed{}."
            text=tokenizer.apply_chat_template([{"role":"user","content":content}],tokenize=False,add_generation_prompt=True)
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
            "teacher":"unsloth/Llama-3.2-3B-Instruct","revision":"006f5dcd1393c3add266de40994ba96225e9689d",
            "target_adapters":None,"records":len(items),"job_id":os.environ.get("SLURM_JOB_ID")})
        print(json.dumps({"split":split,"generated":len(items),"seconds":time.perf_counter()-start}),flush=True)


def capture(out,role):
    import torch
    from vllm import LLM,PoolingParams
    path=MODELS/("llama3-target" if role == "target" else "llama8-source")
    width=3072 if role == "target" else 4096
    taps=[1,7,13,19,25] if role == "target" else [1,8,15,22,29]
    os.environ["AUF_TARGET_TAPS"]=json.dumps(taps)
    llm=LLM(model=str(path),runner="pooling",hf_overrides={"architectures":["AUFCaptureLlama"]},
            pooler_config={"task":"token_embed"},dtype="bfloat16",enforce_eager=True,
            max_model_len=5120,max_num_seqs=4,max_num_batched_tokens=8192,gpu_memory_utilization=.8,
            enable_chunked_prefill=False,enable_prefix_caching=False)
    kwargs=dict(pooling_task="token_embed",pooling_params=PoolingParams(task="token_embed"),use_tqdm=False)
    for split in ["train","dev"]:
        manifest=out/f"{split}.json"
        examples=json.loads(manifest.read_text())
        manifest_hash=sha(manifest)
        for start in range(0,len(examples),4):
            group=examples[start:start+4]
            outputs=llm.encode([{"prompt_token_ids":row["full_ids"]} for row in group],**kwargs)
            for offset,(row,result) in enumerate(zip(group,outputs)):
                features=result.outputs.data.to(torch.bfloat16).cpu()
                assert features.shape == (len(row["full_ids"]),5*width)
                if start == 0 and offset == 0:
                    boundary=len(row["prompt_token_ids"])
                    prefix=llm.encode([{"prompt_token_ids":row["full_ids"][:boundary]}],**kwargs)[0].outputs.data.float().cpu()
                    error=((features[:boundary].float()-prefix).square().sum()/prefix.square().sum()).item()
                    assert error < 1e-6,error
                    write(out/f"features/{role}/{split}/causality.json",{"passed":True,"relative_mse":error})
                dest=out/f"features/{role}/{split}/{start+offset:05d}.pt"
                dest.parent.mkdir(parents=True,exist_ok=True)
                temporary=dest.with_suffix(".part")
                torch.save({"features":features,"group_id":row["group_id"],"layers":taps},temporary)
                temporary.replace(dest)
                write(dest.with_suffix(".json"),{"sha256":sha(dest),"manifest_sha256":manifest_hash,
                                                "role":role,"job_id":os.environ.get("SLURM_JOB_ID")})
            print(json.dumps({"role":role,"split":split,"captured":start+len(group)}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["generate","target","source"])
    parser.add_argument("--out",type=Path,default=ROOT/"pilot-l3")
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.mode == "generate":
        generate(args.out)
    else:
        capture(args.out,args.mode)
