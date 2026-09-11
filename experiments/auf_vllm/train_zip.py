"""Original ZIP feature objective/weighting using the existing paired cache."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
import torch
from safetensors.torch import load_file
from .interfaces import LayerContextMapper
from .pilot_data import write,sha
from .train_pilot import export


def main(args):
    sys.path.insert(0,str(Path(__file__).parent/"runtime_zip"))
    from data import CACHE,pairs,batches,rows
    from sampling import positions
    work=Path(os.environ["TRANSFER_WORK"])
    assert json.loads((CACHE/"manifest.json").read_text())["complete"]
    selected=list(rows("train",args.records))
    if args.manifest is not None:
        expected=json.loads(args.manifest.read_text())[:args.records]
        assert [x["group_id"] for x in selected] == [x["group_id"] for x in expected]
        assert all(a["full_ids"] == b["full_ids"] for a,b in zip(selected,expected))
    args.out.mkdir(parents=True,exist_ok=True)
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    config.update(weighting="equal total example mass",positions="original deterministic segment-stratified 25%",
                  objective="layer relative MSE + normalized fused-context relative MSE")
    write(args.out/"contract.json",config)
    weights=load_file(str(work/"models/4b/draft/model.safetensors"))
    torch.manual_seed(args.seed)
    mapper=LayerContextMapper(weights["fc.weight"],weights["hidden_norm.weight"]).to("cuda")
    optimizer=torch.optim.AdamW(mapper.parameters(),lr=args.lr,weight_decay=0,fused=True)
    total=sum(len(positions(len(x["full_ids"]),len(x["prompt_token_ids"]),x["group_id"])) for x in selected)
    steps_epoch=math.ceil(total/2048)
    steps=steps_epoch*args.epochs
    warm=max(1,int(.05*steps))
    history=[]
    step=0
    start_epoch=0
    resume=args.out/"resume.pt"
    if resume.exists():
        saved=torch.load(resume,weights_only=True)
        assert saved["config"] == config
        mapper.load_state_dict(saved["mapper"])
        optimizer.load_state_dict(saved["optimizer"])
        history,step,start_epoch=saved["history"],saved["step"],saved["epoch"]+1
    start=time.perf_counter()
    for epoch in range(start_epoch,args.epochs):
        train_sum=0.
        epoch_start=time.perf_counter()
        for x,y,w in batches(pairs("train",args.records,4096),2048,args.seed+epoch):
            rate=(step+1)/warm if step<warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,steps-warm)))
            for group in optimizer.param_groups:
                group["lr"]=args.lr*rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss=mapper.feature_loss(x.to("cuda"),y.to("cuda"))
                weighted=(loss*w.to("cuda")).sum()
                objective=weighted*(total/args.records)/2048
            objective.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            train_sum+=weighted.item()
            step+=1
        event={"epoch":epoch+1,"feature_loss":train_sum/args.records,"step":step,
               "seconds":time.perf_counter()-epoch_start,"positions":total,"records":args.records}
        history.append(event)
        temporary=resume.with_suffix(".part")
        torch.save({"config":config,"mapper":mapper.state_dict(),"optimizer":optimizer.state_dict(),
                    "epoch":epoch,"step":step,"history":history},temporary)
        temporary.replace(resume)
        export(mapper,work,args.out/f"epoch-{epoch+1}/export",next(pairs("train",1,4096))[1])
        write(args.out/"history.json",history)
        print(json.dumps(event),flush=True)
    write(args.out/"summary.json",{"config":config,"history":history,"seconds_this_invocation":time.perf_counter()-start,
          "job_id":os.environ.get("SLURM_JOB_ID"),"checkpoint_sha256":sha(resume),"status":"complete"})


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--records",type=int,default=4096)
    parser.add_argument("--epochs",type=int,default=3)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args())
