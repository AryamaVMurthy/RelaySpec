"""Streaming token training with exact microbatch-normalized AUF or CE."""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path
import torch
from .blocks import collate_blocks
from .interfaces import LayerContextMapper
from .losses import token_loss
from .pilot_data import write,sha
from .train_pilot import load_models,block_for,logits,export,batch_gate


class Records:
    def __init__(self,root,split="train",count=None):
        self.root,self.split=root,split
        self.rows=json.loads((root/f"{split}.json").read_text())
        if count is not None:
            assert count <= len(self.rows)
            self.rows=self.rows[:count]
        assert len({x["group_id"] for x in self.rows}) == len(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        captured=torch.load(self.root/f"features/8/{self.split}/{index:05d}.pt",weights_only=True)
        assert captured["group_id"] == row["group_id"]
        assert captured["features"].shape == (len(row["full_ids"]),20480)
        return row,captured["features"],None


@torch.no_grad()
def validation(draft,embedding,mapper,records):
    totals={"ce":0.,"auf":0.,"accepted_prefix":0.,"blocks":0}
    for index in range(len(records)):
        example=records[index]
        row=example[0]
        lo,hi=len(row["prompt_token_ids"]),len(row["full_ids"])-2
        if hi < lo:
            raise ValueError(f"Validation record {index} has no supervised block")
        anchors=sorted(set(torch.linspace(lo,hi,4).long().tolist()))
        for anchor in anchors:
            batch=collate_blocks([block_for(example,anchor,draft)],"cuda")
            with torch.autocast("cuda",dtype=torch.bfloat16):
                scores=logits(draft,embedding,mapper,batch)
                for name in ["ce","auf"]:
                    totals[name]+=token_loss(scores,batch["labels"],batch["valid"],name).loss.item()
            prefix=((~batch["valid"]) | scores.argmax(-1).eq(batch["labels"])).long().cumprod(-1)
            totals["accepted_prefix"]+=(prefix*batch["valid"]).sum().item()
            totals["blocks"]+=1
    return {key:value/totals["blocks"] if key != "blocks" else value for key,value in totals.items()}


def main(args):
    if args.logical_records % args.microbatch_records:
        raise ValueError("Logical batch must be divisible by microbatch records")
    args.out.mkdir(parents=True,exist_ok=True)
    contract={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    contract.update(train_manifest_sha256=sha(args.data/"train.json"),original_frozen_targets=True,
                    normalization="mean active-token CE inside each microbatch; equal microbatch means per update")
    contract_path=args.out/"contract.json"
    if contract_path.exists():
        assert json.loads(contract_path.read_text()) == contract
    else:
        write(contract_path,contract)
    train=Records(args.data,count=args.records)
    if args.defer_validation:
        valid=None
        validation_rows=json.loads((args.validation_data/"dev-prompts.json").read_text())
    else:
        valid=Records(args.validation_data,split=args.validation_split)
        validation_rows=valid.rows
    if not args.exploratory and len(validation_rows) < 1024:
        raise ValueError("Main fitting requires the separate 1024-record offline validation manifest")
    assert not ({x["group_id"] for x in train.rows} & {x["group_id"] for x in validation_rows})
    draft,embedding=load_models(Path(os.environ["TRANSFER_WORK"]))
    torch.manual_seed(args.seed)
    mapper=LayerContextMapper(draft.fc.weight,draft.hidden_norm.weight).to("cuda")
    gate=batch_gate(draft,embedding,mapper,[train[0],train[1]])
    write(args.out/"gate.json",gate)
    optimizer=torch.optim.AdamW(mapper.parameters(),lr=args.lr,weight_decay=0,fused=True)
    planned_steps=math.ceil(len(train)/args.logical_records)*args.epochs
    warm=max(1,int(.05*planned_steps))
    start_epoch=step=0
    history=[]
    resume=args.out/"resume.pt"
    if resume.exists():
        saved=torch.load(resume,weights_only=True)
        assert saved["contract"] == contract
        mapper.load_state_dict(saved["mapper"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch,step,history=saved["epoch"]+1,saved["step"],saved["history"]
    frozen_versions={name:p._version for name,p in draft.named_parameters()}
    torch.cuda.reset_peak_memory_stats()
    invocation=time.perf_counter()
    for epoch in range(start_epoch,args.epochs):
        rng=random.Random(args.seed+epoch)
        order=list(range(len(train)))
        rng.shuffle(order)
        visited=[]
        total_blocks=total_active=total_valid=0
        loss_sum=0.
        epoch_start=time.perf_counter()
        for start in range(0,len(order),args.logical_records):
            logical=order[start:start+args.logical_records]
            micro_count=math.ceil(len(logical)/args.microbatch_records)
            optimizer.zero_grad(set_to_none=True)
            rate=(step+1)/warm if step<warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,planned_steps-warm)))
            for group in optimizer.param_groups:
                group["lr"]=args.lr*rate
            for offset in range(0,len(logical),args.microbatch_records):
                blocks=[]
                for index in logical[offset:offset+args.microbatch_records]:
                    example=train[index]
                    row=example[0]
                    candidates=list(range(len(row["prompt_token_ids"]),len(row["full_ids"])-1))
                    if not candidates:
                        raise ValueError(f"Training record {index} has no supervised block")
                    anchors=rng.sample(candidates,min(4,len(candidates)))
                    blocks.extend(block_for(example,a,draft) for a in anchors)
                    visited.append(index)
                batch=collate_blocks(blocks,"cuda")
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    scores=logits(draft,embedding,mapper,batch)
                    result=token_loss(scores,batch["labels"],batch["valid"],args.objective)
                (result.loss/micro_count).backward()
                total_blocks+=len(blocks)
                total_active+=int(result.active.sum())
                total_valid+=int(batch["valid"].sum())
                loss_sum+=result.loss.item()/micro_count
                del blocks,batch,scores,result,example
            norm=torch.nn.utils.clip_grad_norm_(mapper.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            step+=1
            if step % 16 == 0:
                print(json.dumps({"epoch":epoch+1,"step":step,"records_visited":len(visited),
                    "seconds":time.perf_counter()-epoch_start,"gradient_norm":float(norm)}),flush=True)
        assert sorted(visited) == list(range(len(train)))
        event={"epoch":epoch+1,"step":step,"records_visited":len(visited),"blocks":total_blocks,
               "active_tokens":total_active,"valid_tokens":total_valid,
               "mean_update_loss":loss_sum/math.ceil(len(train)/args.logical_records),
               "training_seconds":time.perf_counter()-epoch_start,
               "validation":validation(draft,embedding,mapper,valid) if valid is not None else {"status":"pending_offline_evaluation"}}
        history.append(event)
        temporary=resume.with_suffix(".part")
        torch.save({"contract":contract,"mapper":mapper.state_dict(),"optimizer":optimizer.state_dict(),
                    "epoch":epoch,"step":step,"history":history},temporary)
        temporary.replace(resume)
        write(args.out/"history.json",history)
        export(mapper,Path(os.environ["TRANSFER_WORK"]),args.out/f"epoch-{epoch+1}/export",train[0][1])
        print(json.dumps(event),flush=True)
    assert all(p.grad is None and p._version == frozen_versions[name] for name,p in draft.named_parameters())
    write(args.out/"summary.json",{"contract":contract,"history":history,"job_id":os.environ.get("SLURM_JOB_ID"),
          "seconds_this_invocation":time.perf_counter()-invocation,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
          "checkpoint_sha256":sha(resume),"status":"training_complete_validation_pending" if valid is None else "complete"})


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--validation-data",type=Path,required=True)
    parser.add_argument("--validation-split",default="dev")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--objective",choices=["auf","ce"],required=True)
    parser.add_argument("--records",type=int,default=4096)
    parser.add_argument("--epochs",type=int,default=3)
    parser.add_argument("--microbatch-records",type=int,default=4)
    parser.add_argument("--logical-records",type=int,default=32)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--exploratory",action="store_true")
    parser.add_argument("--defer-validation",action="store_true",help="Fit fixed epochs while separately reserved validation labels are prepared")
    main(parser.parse_args())
