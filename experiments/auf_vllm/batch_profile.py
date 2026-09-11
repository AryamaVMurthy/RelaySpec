"""Measure exact AUF forward/backward at candidate record microbatches."""
import argparse
import json
import os
import time
from pathlib import Path
import torch
from .blocks import collate_blocks
from .interfaces import LayerContextMapper
from .losses import token_loss
from .pilot_data import write
from .train_pilot import load_models, block_for, logits


def main(args):
    draft, embedding = load_models(Path(os.environ["TRANSFER_WORK"]))
    torch.manual_seed(42)
    mapper = LayerContextMapper(draft.fc.weight,draft.hidden_norm.weight).to("cuda")
    rows = json.loads((args.data/"train.json").read_text())
    order = sorted(range(len(rows)),key=lambda i:len(rows[i]["full_ids"]),reverse=True)[:8]
    examples=[]
    for index in order:
        captured=torch.load(args.data/f"features/8/train/{index:05d}.pt",weights_only=True)
        assert captured["group_id"] == rows[index]["group_id"]
        examples.append((rows[index],captured["features"],None))
    results=[]
    for records in [1,2,4,8]:
        blocks=[]
        for example in examples[:records]:
            row=example[0]
            lo=len(row["prompt_token_ids"])
            hi=len(row["full_ids"])-2
            anchors=torch.linspace(lo,hi,4).long().tolist()
            blocks.extend(block_for(example,a,draft) for a in anchors)
        batch=None
        scores=None
        loss=None
        try:
            batch=collate_blocks(blocks,"cuda")
            torch.cuda.reset_peak_memory_stats()
            times=[]
            for repeat in range(3):
                mapper.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                start=time.perf_counter()
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    scores=logits(draft,embedding,mapper,batch)
                    loss=token_loss(scores,batch["labels"],batch["valid"],"auf")
                loss.loss.backward()
                torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
            results.append({"records":records,"anchors_per_record":4,"seconds":times,
                "warmed_records_per_second":records/(sum(times[1:])/2),
                "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"status":"passed"})
        except torch.cuda.OutOfMemoryError as error:
            results.append({"records":records,"status":"out_of_memory","error":str(error)})
        finally:
            del batch,scores,loss,blocks
            mapper.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        print(json.dumps(results[-1]),flush=True)
        write(args.out,{"job_id":os.environ.get("SLURM_JOB_ID"),"device":torch.cuda.get_device_name(),
                        "selection":"eight longest pilot records; four anchors spanning response", "results":results})


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args())
