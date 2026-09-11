"""Freeze the offline validation split and batch original-target rollouts."""
import argparse
import json
import os
import time
from pathlib import Path
from .pilot_data import rows,sha,write


def main(args):
    from vllm import LLM,SamplingParams
    work=Path(os.environ["TRANSFER_WORK"])
    args.out.mkdir(parents=True,exist_ok=True)
    source=work/"common/dataset/eval.jsonl"
    # Preserve the published first 128 examples as exposed legacy benchmarks.
    validation=rows(source)[128:1152]
    excluded=rows(work/"common/dataset/train.jsonl")+rows(work/"common/dataset/dev.jsonl")
    assert len(validation) == 1024
    assert not ({x["group_id"] for x in validation} & {x["group_id"] for x in excluded})
    manifest=args.out/"dev-prompts.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == validation
    else:
        write(manifest,validation)
    write(args.out/"provenance.json",{"source":str(source),"source_sha256":sha(source),
          "indices":[128,1152],"purpose":"offline validation, excluded from confirmation",
          "exposure":"previously reserved ZIP evaluation pool; no claim of fresh confirmation",
          "teacher":str((work/"models/8b/target").resolve()),"target_adapters":None,
          "records":1024,"output_cap":4096,"prompt_cap":1024,
          "active_sequences":64,"batched_tokens":8192})
    llm=LLM(model=str(work/"models/8b/target"),dtype="bfloat16",max_model_len=5120,
            max_num_seqs=64,max_num_batched_tokens=8192,gpu_memory_utilization=.8,
            enable_prefix_caching=False,seed=42,attention_config={"backend":"FLASH_ATTN"},
            generation_config="vllm",compilation_config={"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"})
    params=SamplingParams(temperature=0,max_tokens=4096,seed=42,stop_token_ids=[151645])
    assembled=[]
    for start in range(0,len(validation),128):
        dest=args.out/f"rollouts/{start:05d}.json"
        group=validation[start:start+128]
        if dest.exists():
            result=json.loads(dest.read_text())
            assert [x["group_id"] for x in result] == [x["group_id"] for x in group]
        else:
            began=time.perf_counter()
            responses=llm.generate([{"prompt_token_ids":x["prompt_token_ids"]} for x in group],params,use_tqdm=False)
            result=[]
            for row,response in zip(group,responses):
                ids=list(response.outputs[0].token_ids)
                result.append({"row_id":row["row_id"],"group_id":row["group_id"],
                    "prompt_token_ids":row["prompt_token_ids"],"output_ids":ids,
                    "full_ids":row["prompt_token_ids"]+ids,"temperature":0,"thinking":False,
                    "finish_reason":response.outputs[0].finish_reason})
            write(dest,result)
            write(dest.with_suffix(".meta.json"),{"sha256":sha(dest),"seconds":time.perf_counter()-began,
                "output_tokens":sum(len(x["output_ids"]) for x in result),"job_id":os.environ.get("SLURM_JOB_ID")})
            print(json.dumps({"validation_rollouts_done":start+len(result),"seconds":time.perf_counter()-began}),flush=True)
        assembled.extend(result)
    write(args.out/"dev.json",assembled)
    # Capture helper expects both splits. The training split here is empty.
    write(args.out/"train.json",[])


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args())
