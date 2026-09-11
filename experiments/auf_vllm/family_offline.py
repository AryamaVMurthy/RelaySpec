"""Fixed 1024-record, four-anchor offline validation for family transfers."""
import argparse
import json
import os
from pathlib import Path
import torch
from safetensors import safe_open
from .family_training import FamilyRecords, FamilyRuntime
from .evaluate_offline import FoldedInterface
from .blocks import collate_blocks
from .losses import token_loss
from .pilot_data import sha, write


@torch.no_grad()
def main(args):
    records=FamilyRecords(args.data,split='dev')
    assert len(records)==1024
    if args.manifest_only:
        write(args.out,records.rows)
        return
    runtime=FamilyRuntime(args.models,args.family)
    results=[]
    for epoch in [1,3]:
        export=args.fit/f'epoch-{epoch}/export/model.safetensors'
        with safe_open(export,framework='pt',device='cpu') as reader:
            mapper=FoldedInterface(reader.get_tensor('fc.weight'),reader.get_tensor('hidden_norm.weight')).to('cuda')
        totals={'ce':0.,'auf':0.,'accepted_prefix':0.,'blocks':0}
        for index in range(len(records)):
            row,x=records.get(index)
            lo,hi=len(row['prompt_token_ids']),len(row['full_ids'])-2
            assert hi>=lo
            for anchor in sorted(set(torch.linspace(lo,hi,4).long().tolist())):
                batch=collate_blocks([runtime.block(row,x,anchor)],'cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    scores=runtime.logits(mapper,batch)
                    for loss in ['ce','auf']:
                        totals[loss]+=token_loss(scores,batch['labels'],batch['valid'],loss).loss.item()
                prefix=((~batch['valid']) | scores.argmax(-1).eq(batch['labels'])).long().cumprod(-1)
                totals['accepted_prefix']+=(prefix*batch['valid']).sum().item()
                totals['blocks']+=1
        metrics={k:v/totals['blocks'] if k!='blocks' else v for k,v in totals.items()}
        results.append({'epoch':epoch,'metrics':metrics,'export_sha256':sha(export)})
        write(args.fit/'offline-validation.json',{'records':len(records),'manifests':records.manifests,
              'results':results,'job_id':os.environ.get('SLURM_JOB_ID'),'measurement':'offline frozen teacher contexts, not decoding throughput'})
        print(json.dumps(results[-1]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--family',choices=['llama','q14'])
    parser.add_argument('--models',type=Path)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--fit',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--manifest-only',action='store_true')
    main(parser.parse_args())
