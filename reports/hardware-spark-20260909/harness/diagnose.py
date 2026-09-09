"""One bounded identical-prefix audit of a native/AR disagreement on GB10."""
import argparse
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"source"))


def main():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from relayspec.dflash import import_official_dflash
    from relayspec.generation import native_autoregressive_generate
    from relayspec.benchmarking import benchmark_turns
    from run import MODELS,sha
    p=argparse.ArgumentParser();p.add_argument("--reference",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    done=json.loads((args.reference/"complete.json").read_text());assert done["status"]=="pass"
    rows=[r for r in map(json.loads,(args.reference/"evaluation.jsonl").read_text().splitlines()) if r["repeat"]==0]
    lookup={(r["method"],r["problem_id"]):r for r in rows}
    records=json.loads((args.reference/"records.json").read_text())
    record=next(r for r in records if lookup["ar",r["problem_id"]]["tokens"]!=lookup["native",r["problem_id"]]["tokens"])
    pid=record["problem_id"];a=lookup["ar",pid]["tokens"];b=lookup["native",pid]["tokens"]
    index=next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y)
    cap=max(64,index+2);assert cap<=256,"Selected mismatch is outside this bounded diagnostic"
    torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
    cache=os.environ["HF_HUB_CACHE"]
    load=dict(cache_dir=cache,local_files_only=True,dtype=torch.bfloat16,attn_implementation="sdpa")
    target=AutoModelForCausalLM.from_pretrained(MODELS["target"][0],revision=MODELS["target"][1],**load).cuda().eval().requires_grad_(False)
    Draft,generate=import_official_dflash(os.environ["DFLASH_SOURCE"],json.loads((HERE/"protocol.json").read_text())["official_commit"])
    draft=Draft.from_pretrained(MODELS["native"][0],revision=MODELS["native"][1],**load).cuda().eval().requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(MODELS["target"][0],revision=MODELS["target"][1],cache_dir=cache,local_files_only=True)
    ids=tokenizer.apply_chat_template([{"role":"user","content":benchmark_turns(record)[0]}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt")
    ids=(ids["input_ids"] if hasattr(ids,"keys") else ids).cuda()
    assert ids.shape[1]==lookup["ar",pid]["input_tokens"]
    desired_position=ids.shape[1]+index-1
    common_prefix=torch.cat((ids[0].cpu(),torch.tensor(a[:index],dtype=torch.long)))
    original=target.forward;active=[""];captured={"ar":[],"native":[]}
    def forward(*positional,**kwargs):
        inputs=positional[0] if positional else kwargs["input_ids"]
        cache_state=kwargs.get("past_key_values")
        start=cache_state.get_seq_length() if cache_state is not None else 0
        kwargs["output_hidden_states"]=True
        result=original(*positional,**kwargs)
        offset=desired_position-start
        logit_offset=offset-(inputs.shape[1]-result.logits.shape[1])
        if 0<=offset<inputs.shape[1] and logit_offset>=0:
            prefix=inputs[0,:offset+1].detach().cpu()
            if torch.equal(prefix,common_prefix[start:desired_position+1]):
                captured[active[0]].append({"cache_length_before":int(start),"input_length":inputs.shape[1],
                    "logits":result.logits[0,logit_offset].detach().float().cpu(),
                    "hidden":result.hidden_states[-1][0,offset].detach().cpu()})
        return result
    target.forward=forward
    results={}
    with torch.inference_mode():
        for method in ["ar","native"]:
            active[0]=method
            common=dict(input_ids=ids,max_new_tokens=cap,stop_token_ids=[tokenizer.eos_token_id],temperature=0.,return_stats=True)
            value=native_autoregressive_generate(target,**common) if method=="ar" else generate(draft,target=target,block_size=16,**common)
            tokens=value.output_ids[0,ids.shape[1]:].tolist()
            assert tokens[:index+1]==lookup[method,pid]["tokens"][:index+1],"Probe did not reproduce the measured disagreement"
            assert len(captured[method])==1,captured[method]
            results[method]=tokens
        target.forward=original
        head=target.lm_head.weight.float()
        exported={}
        for method in ["ar","native"]:
            entry=captured[method][0]
            full32=torch.nn.functional.linear(entry["hidden"].cuda().float().unsqueeze(0),head)[0].cpu()
            def top(logits):
                values,indices=logits.topk(8)
                return [{"id":int(i),"text":tokenizer.decode([int(i)]),"logit":float(v)} for i,v in zip(indices,values)]
            exported[method]={"cache_length_before":entry["cache_length_before"],"input_length":entry["input_length"],
                "measured_bf16_argmax":int(entry["logits"].argmax()),"bf16_top8":top(entry["logits"]),
                "fp32_head_argmax":int(full32.argmax()),"fp32_head_top8":top(full32)}
        hidden_delta=(captured["ar"][0]["hidden"].float()-captured["native"][0]["hidden"].float()).abs()
        result={"status":"pass","problem_id":pid,"first_differing_generated_token_index":index,"absolute_prediction_position":desired_position,
            "identical_token_prefix":True,"reproduced_measured_disagreement":True,"diagnostic_output_cap":cap,
            "hidden_max_abs_difference":float(hidden_delta.max()),"hidden_mean_abs_difference":float(hidden_delta.mean()),
            "fp32_head_argmax_agrees":exported["ar"]["fp32_head_argmax"]==exported["native"]["fp32_head_argmax"],
            "methods":exported,"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),
            "reference_sha256":sha(args.reference/"evaluation.jsonl"),"script_sha256":sha(__file__),
            "scope":"One observed native/AR discrepancy, independent of RelaySpec. Identical committed token prefix; AR and block evaluation use different matrix shapes/cache histories. FP32 head reprojection reuses saved BF16 hidden states; it is not full-FP32 inference or a universal exactness guarantee."}
        torch.save({"captured":captured,"probe_tokens":results},args.output/"captured-tensors.pt")
        (args.output/"result.json").write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
