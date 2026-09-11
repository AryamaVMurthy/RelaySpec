"""Same-family Llama ZIP/CE/AUF pilot; original untied source head retained."""
import argparse
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from torch.nn import functional as F
from .blocks import make_block,collate_blocks,conditioned_forward
from .interfaces import LayerContextMapper
from .losses import token_loss
from .pilot_data import write,sha
from .train_pilot import tensor


ROOT=Path("/scratch/aryama.murthy/relayspec-auf-20260911")


def main(args):
    from transformers import Qwen3Config
    sys.path.insert(0,os.environ["DFLASH_SOURCE"])
    cls=importlib.import_module("dflash.model").DFlashDraftModel
    models=ROOT/"models"
    is_llama=args.family == "llama"
    source_name,target_name,draft_name=("llama8-source","llama3-target","llama8-draft") if is_llama else ("qwen4-source","qwen14-target","qwen4-draft")
    source_width,target_width=(4096,3072) if is_llama else (2560,5120)
    taps=[1,7,13,19,25] if is_llama else [1,10,19,28,37]
    path=models/draft_name
    cfg=Qwen3Config.from_pretrained(path,local_files_only=True)
    cfg._attn_implementation="sdpa"
    draft=cls(cfg)
    draft.load_state_dict(load_file(str(path/"model.safetensors")),strict=True,assign=True)
    draft=draft.to("cuda",torch.bfloat16).eval().requires_grad_(False)
    embedding=tensor(models/source_name,"model.embed_tokens.weight").to("cuda",torch.bfloat16)
    head=tensor(models/source_name,"lm_head.weight").to("cuda",torch.bfloat16) if is_llama else embedding
    assert embedding.shape == head.shape and embedding.shape[1] == source_width
    if is_llama:
        assert not torch.equal(embedding[:8],head[:8])
    target_cfg=json.loads((models/target_name/"config.json").read_text())
    eos=target_cfg["eos_token_id"]
    torch.manual_seed(42)
    mapper=LayerContextMapper(draft.fc.weight,draft.hidden_norm.weight,target_width=target_width,source_width=source_width).to("cuda")
    data=ROOT/("pilot-l3" if is_llama else "pilot-q14")
    parts={}
    for split in ["train","dev"]:
        examples=[]
        for index,row in enumerate(json.loads((data/f"{split}.json").read_text())):
            captured=[]
            for role in ["target","source"]:
                item=torch.load(data/f"features/{role}/{split}/{index:05d}.pt",weights_only=True)
                assert item["group_id"] == row["group_id"]
                captured.append(item["features"])
            examples.append((row,*captured))
        parts[split]=examples

    def block(example,anchor):
        row,x,_=example
        return make_block(x,row["full_ids"],anchor,draft.block_size,draft.mask_token_id,(eos,))

    def forward(batch):
        context,_=mapper(batch["context"])
        noise=F.embedding(batch["noise_ids"],embedding)
        hidden=conditioned_forward(draft,context,noise,batch["position_ids"],batch["attention_mask"])
        return F.linear(hidden,head)

    @torch.no_grad()
    def evaluate():
        values={"ce":0.,"auf":0.,"feature":0.,"accepted_prefix":0.}
        with torch.autocast("cuda",dtype=torch.bfloat16):
            for example in parts["dev"]:
                row,x,y=example
                batch=collate_blocks([block(example,len(row["prompt_token_ids"]))],"cuda")
                scores=forward(batch)
                for objective in ["ce","auf"]:
                    values[objective]+=token_loss(scores,batch["labels"],batch["valid"],objective).loss.item()
                prefix=((~batch["valid"]) | scores.argmax(-1).eq(batch["labels"])).long().cumprod(-1)
                values["accepted_prefix"]+=(prefix*batch["valid"]).sum().item()
                positions=torch.linspace(0,len(x)-1,min(32,len(x))).long()
                values["feature"]+=mapper.feature_loss(x[positions].to("cuda"),y[positions].to("cuda")).mean().item()
        return {key:value/len(parts["dev"]) for key,value in values.items()}

    args.out.mkdir(parents=True,exist_ok=True)
    initial=evaluate()
    optimizer=torch.optim.AdamW(mapper.parameters(),lr=1e-4,weight_decay=0,fused=True)
    rng=random.Random(42)
    history=[{"epoch":0,"validation":initial}]
    start=time.perf_counter()
    for epoch in range(args.epochs):
        order=list(range(len(parts["train"])))
        rng.shuffle(order)
        total=0.
        for index in order:
            example=parts["train"][index]
            row,x,y=example
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                if args.objective == "zip":
                    positions=torch.randperm(len(x))[:32]
                    loss=mapper.feature_loss(x[positions].to("cuda"),y[positions].to("cuda")).mean()
                else:
                    anchor=rng.randrange(len(row["prompt_token_ids"]),len(row["full_ids"])-1)
                    batch=collate_blocks([block(example,anchor)],"cuda")
                    scores=forward(batch)
                    loss=token_loss(scores,batch["labels"],batch["valid"],args.objective).loss
            loss.backward()
            assert all(p.grad is None and not p.requires_grad for p in draft.parameters())
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in mapper.parameters())
            torch.nn.utils.clip_grad_norm_(mapper.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            total+=loss.item()
        event={"epoch":epoch+1,"training_loss":total/len(order),"validation":evaluate(),"seconds":time.perf_counter()-start}
        history.append(event)
        write(args.out/"history.json",history)
        print(json.dumps(event),flush=True)
    torch.save({"mapper":mapper.state_dict(),"objective":args.objective,"seed":42},args.out/"checkpoint.pt")
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        folded=mapper.folded().to(torch.bfloat16)
        x=parts["train"][0][1][:256].to("cuda")
        expected=mapper(x)[0]
        actual=mapper.normalize(F.linear(x,folded))
        error=((actual.float()-expected.float()).square().sum(-1)/(expected.float().square().sum(-1)+1e-6)).mean().item()
        assert error < 1e-3,error
    weights=load_file(str(path/"model.safetensors"))
    weights.update({"fc.weight":folded.detach().cpu().contiguous(),"embed_tokens.weight":embedding.cpu().clone(),"lm_head.weight":head.cpu().clone()})
    export=args.out/"export"
    export.mkdir(exist_ok=True)
    save_file(weights,str(export/"model.safetensors"))
    config=json.loads((path/"config.json").read_text())
    config.update(architectures=["MapperDFlash"],target_hidden_size=target_width,num_target_layers=target_cfg["num_hidden_layers"],
                  tie_word_embeddings=False,bos_token_id=target_cfg["bos_token_id"],eos_token_id=eos)
    config["dflash_config"]["target_layer_ids"]=taps
    write(export/"config.json",config)
    write(export/"export_check.json",{"context_relative_mse":error,"source_head_untied":is_llama})
    write(args.out/"summary.json",{"pilot_only":True,"objective":args.objective,"records":32,"epochs":args.epochs,
          "source":source_name,"target":target_name,"target_adapters":None,"source_head_untied":is_llama,
          "initial":initial,"final":history[-1]["validation"],"seconds":time.perf_counter()-start,
          "job_id":os.environ.get("SLURM_JOB_ID"),"checkpoint_sha256":sha(args.out/"checkpoint.pt")})


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--objective",choices=["auf","ce","zip"],required=True)
    parser.add_argument("--family",choices=["llama","q14"],default="llama")
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args())
