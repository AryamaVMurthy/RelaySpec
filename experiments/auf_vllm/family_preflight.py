"""Llama model/vocabulary and vLLM auxiliary-capture compatibility gate."""
import argparse
import json
import os
from pathlib import Path
import torch
from .pilot_data import write,sha


def main(args):
    from transformers import AutoTokenizer
    from vllm import LLM,PoolingParams,SamplingParams
    root=Path("/scratch/aryama.murthy/relayspec-auf-20260911/models")
    target=root/"llama3-target"
    source=root/"llama8-source"
    draft=root/"llama8-draft"
    tt=AutoTokenizer.from_pretrained(target,local_files_only=True)
    st=AutoTokenizer.from_pretrained(source,local_files_only=True)
    samples=["Hello, world!","def f(x):\n    return x + 1\n","café 日本語 😀", "  12.5\n\n"]
    shared=tt.get_vocab() == st.get_vocab()
    assert shared and all(tt.encode(x,add_special_tokens=False)==st.encode(x,add_special_tokens=False) for x in samples)
    ids=tt.apply_chat_template([{"role":"user","content":"What is 2 + 2? Answer with only the number."}],tokenize=True,add_generation_prompt=True)
    info={"job_id":os.environ.get("SLURM_JOB_ID"),"gpu":torch.cuda.get_device_name(),
          "shared_token_vocab":shared,"vocab_size":len(tt),"target_adapters":None,
          "configs":{name:{"sha256":sha(path/"config.json"),"config":json.loads((path/"config.json").read_text())}
                     for name,path in [("target",target),("source",source),("draft",draft)]}}
    if args.mode == "generate":
        llm=LLM(model=str(target),dtype="bfloat16",enforce_eager=True,max_model_len=5120,
                max_num_seqs=4,max_num_batched_tokens=8192,gpu_memory_utilization=.7,
                enable_prefix_caching=False,generation_config="vllm")
        output=llm.generate([{"prompt_token_ids":ids}],SamplingParams(temperature=0,max_tokens=32),use_tqdm=False)[0].outputs[0]
        info.update(text=output.text,output_ids=list(output.token_ids),passed=output.text.strip()=="4")
        assert info["passed"],info
    else:
        llm=LLM(model=str(target),runner="pooling",hf_overrides={"architectures":["AUFCaptureLlama"]},
                pooler_config={"task":"token_embed"},dtype="bfloat16",enforce_eager=True,
                max_model_len=5120,max_num_seqs=4,max_num_batched_tokens=8192,gpu_memory_utilization=.7,
                enable_prefix_caching=False,enable_chunked_prefill=False)
        full=ids+tt.encode("4. This is an extra suffix for the causal feature test.",add_special_tokens=False)
        kwargs=dict(pooling_task="token_embed",pooling_params=PoolingParams(task="token_embed"),use_tqdm=False)
        outputs=llm.encode([{"prompt_token_ids":full},{"prompt_token_ids":ids}],**kwargs)
        a,b=(x.outputs.data.float().cpu() for x in outputs)
        assert a.shape == (len(full),5*3072) and b.shape == (len(ids),5*3072)
        relative=((a[:len(ids)]-b).square().sum()/b.square().sum()).item()
        assert relative < 1e-6,relative
        info.update(passed=True,causality_relative_mse=relative,target_taps=[1,7,13,19,25])
    write(args.out,info)
    print(json.dumps({"mode":args.mode,"passed":info["passed"]}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["generate","capture"])
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args())
