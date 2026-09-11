"""Short Llama vLLM compatibility diagnostic; not a paper timing endpoint."""
import argparse
import json
import os
import time
from pathlib import Path
from .pilot_data import write


def main(args):
    from vllm import LLM,SamplingParams
    from transformers import AutoTokenizer
    root=Path("/scratch/aryama.murthy/relayspec-auf-20260911")
    target=root/"models/llama3-target"
    if args.mode != "ar":
        os.environ["TRANSFER_MAPPED"]="1"
        from .runtime_zip.mapper_runtime import install
        install()
    config=dict(model=str(target),dtype="bfloat16",max_model_len=5120,max_num_seqs=4,
                max_num_batched_tokens=8192,gpu_memory_utilization=.8,enable_prefix_caching=False,
                generation_config="vllm",async_scheduling=False,
                compilation_config={"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"})
    if args.mode != "ar":
        assert args.export is not None
        config["speculative_config"]={"method":"dflash","model":str(args.export),"num_speculative_tokens":9}
    tokenizer=AutoTokenizer.from_pretrained(target,local_files_only=True)
    params=SamplingParams(temperature=0,max_tokens=128,seed=0,stop_token_ids=[tokenizer.eos_token_id])
    llm=LLM(**config)
    for prompt in ["What is 2 + 2?","Write a short greeting.","Count to five.","Define a prime number."]:
        text=tokenizer.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)
        llm.generate([{"prompt_token_ids":tokenizer.encode(text,add_special_tokens=False)}],params,use_tqdm=False)
    rows=json.loads((root/"pilot-l3/dev.json").read_text())[:4]
    results=[]
    for row in rows:
        start=time.perf_counter()
        output=llm.generate([{"prompt_token_ids":row["prompt_token_ids"]}],params,use_tqdm=False)[0].outputs[0]
        results.append({"group_id":row["group_id"],"output_ids":list(output.token_ids),"text":output.text,
                        "seconds":time.perf_counter()-start,"finish_reason":output.finish_reason})
    write(args.out,{"mode":args.mode,"diagnostic_only":True,"count":4,"output_cap":128,
          "timing_note":"warmed diagnostic, not accepted as final paper timing", "rows":results,
          "tps":sum(len(x["output_ids"]) for x in results)/sum(x["seconds"] for x in results),
          "job_id":os.environ.get("SLURM_JOB_ID"),"config":config})
    print(json.dumps({"mode":args.mode,"count":len(results)}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["ar","auf","ce","zip"],required=True)
    parser.add_argument("--export",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args())
