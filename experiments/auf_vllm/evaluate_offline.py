"""Deferred offline token validation of immutable folded deployment checkpoints."""
import argparse
import json
import os
from pathlib import Path
import torch
from safetensors import safe_open
from torch.nn import functional as F
from .pilot_data import write,sha
from .train_pilot import load_models
from .train_tokens import Records,validation


class FoldedInterface(torch.nn.Module):
    def __init__(self,weight,norm):
        super().__init__()
        self.register_buffer("weight",weight)
        self.register_buffer("norm",norm)

    def forward(self,x):
        hidden=F.linear(x,self.weight)
        normalized=(hidden.float()*torch.rsqrt(hidden.float().square().mean(-1,keepdim=True)+1e-6)).to(hidden.dtype)
        return normalized*self.norm,None


def main(args):
    draft,embedding=load_models(Path(os.environ["TRANSFER_WORK"]))
    records=Records(args.data,"dev")
    assert len(records) == 1024
    result=[]
    for epoch in [1,3]:
        path=args.fit/f"epoch-{epoch}/export/model.safetensors"
        with safe_open(path,framework="pt",device="cpu") as reader:
            interface=FoldedInterface(reader.get_tensor("fc.weight"),reader.get_tensor("hidden_norm.weight")).to("cuda")
        event={"epoch":epoch,"export_sha256":sha(path),"metrics":validation(draft,embedding,interface,records),
               "interface":"BF16 folded deployment matrix","job_id":os.environ.get("SLURM_JOB_ID")}
        result.append(event)
        report={"manifest_sha256":sha(args.data/"dev.json"),"records":1024,"results":result}
        write(args.fit/"offline-validation.json",report)
        if args.fit.parent.name == "lr-screen":
            control=Path(os.environ.get("AUF_CONTROL_ROOT","/home/aryama.murthy/relayspec-auf-20260911"))
            write(control/"outputs/lr-screen"/args.fit.name/"offline-validation.json",report)
        print(json.dumps(event),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--fit",type=Path,required=True)
    parser.add_argument("--data",type=Path,required=True)
    main(parser.parse_args())
