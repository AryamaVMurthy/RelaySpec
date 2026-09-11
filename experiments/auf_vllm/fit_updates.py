"""Fixed-update AUF/CE curves, resumable in short GPU chunks."""
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
from .pilot_data import sha,write
from .train_tokens import Records
from .train_pilot import load_models,block_for,logits,export,batch_gate


def record_indices(count,step,seed,batch_size=32):
    """Exactly32 visits/update, including repeated passes for N<32."""
    assert count>0 and step>=0
    cursor=step*batch_size
    result=[]
    while len(result)<batch_size:
        epoch,offset=divmod(cursor,count)
        order=list(range(count));random.Random(seed+epoch).shuffle(order)
        chosen=order[offset:offset+batch_size-len(result)]
        result.extend(chosen);cursor+=len(chosen)
    return result


def main(args):
    records=Records(args.data,count=args.records)
    validation=json.loads((args.validation_data/'dev.json').read_text())
    assert len(validation)==1024
    assert not ({r['group_id'] for r in records.rows}&{r['group_id'] for r in validation})
    args.out.mkdir(parents=True,exist_ok=True)
    contract={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='chunk_updates'}
    contract.update(training_manifest_sha256=sha(args.data/'train.json'),validation_manifest_sha256=sha(args.validation_data/'dev.json'),
                    target_adapters=None,record_visits_per_update=32,anchors_per_visit=4,microbatch_records=4,
                    objective='microbatch-normalized '+args.objective,small_data_policy='cycle shuffled passes; retain32 visits per update',
                    code_sha256={name:sha(Path(__file__).parent/name) for name in ['losses.py','blocks.py','interfaces.py','fit_updates.py']})
    if (args.out/'contract.json').exists():assert json.loads((args.out/'contract.json').read_text())==contract
    else:write(args.out/'contract.json',contract)
    draft,embedding=load_models(Path(os.environ['TRANSFER_WORK']))
    torch.manual_seed(args.seed)
    mapper=LayerContextMapper(draft.fc.weight,draft.hidden_norm.weight).to('cuda')
    optimizer=torch.optim.AdamW(mapper.parameters(),lr=args.lr,weight_decay=0,fused=True)
    resume=args.out/'resume.pt';step=0;history=[]
    if resume.exists():
        state=torch.load(resume,weights_only=True)
        assert state['contract']==contract
        mapper.load_state_dict(state['mapper']);optimizer.load_state_dict(state['optimizer'])
        step,history=state['step'],state['history']
    else:
        assert len(records)>=2
        write(args.out/'gate.json',batch_gate(draft,embedding,mapper,[records[0],records[1]]))
    if step==args.updates:return
    end=min(args.updates,step+args.chunk_updates)
    warmup=max(1,int(.05*args.updates));start_step=step;start=time.perf_counter()
    active=valid=blocks=0;loss_sum=0.
    frozen={n:p._version for n,p in draft.named_parameters()}
    torch.cuda.reset_peak_memory_stats()
    while step<end:
        indices=record_indices(len(records),step,args.seed)
        rng=random.Random(args.seed*1000003+step)
        rate=(step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,args.updates-warmup)))
        for group in optimizer.param_groups:group['lr']=args.lr*rate
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0,32,4):
            batch_blocks=[]
            for index in indices[offset:offset+4]:
                example=records[index];row=example[0]
                positions=range(len(row['prompt_token_ids']),len(row['full_ids'])-1)
                assert len(positions)>=4,'Fewer than four distinct supervised anchors'
                batch_blocks.extend(block_for(example,a,draft) for a in rng.sample(positions,4))
            batch=collate_blocks(batch_blocks,'cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                scores=logits(draft,embedding,mapper,batch)
                result=token_loss(scores,batch['labels'],batch['valid'],args.objective)
            (result.loss/8).backward()
            loss_sum+=result.loss.item()/8
            active+=int(result.active.sum());valid+=int(batch['valid'].sum());blocks+=len(batch_blocks)
            del example,batch_blocks,batch,scores,result
        torch.nn.utils.clip_grad_norm_(mapper.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();step+=1
        if step%16==0:print(json.dumps({'step':step,'seconds':time.perf_counter()-start}),flush=True)
    assert all(p.grad is None and p._version==frozen[n] for n,p in draft.named_parameters())
    event={'start_step':start_step,'step':step,'record_visits':(step-start_step)*32,'total_record_visits':step*32,
           'equivalent_passes':step*32/len(records),'active_tokens':active,'valid_tokens':valid,'blocks':blocks,
           'mean_update_loss':loss_sum/(step-start_step),'training_seconds':time.perf_counter()-start,
           'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'job_id':os.environ.get('SLURM_JOB_ID')}
    history.append(event)
    temporary=resume.with_suffix('.part')
    torch.save({'contract':contract,'mapper':mapper.state_dict(),'optimizer':optimizer.state_dict(),
                'step':step,'history':history},temporary);temporary.replace(resume)
    export(mapper,Path(os.environ['TRANSFER_WORK']),args.out/f'step-{step}/export',records[0][1])
    write(args.out/'history.json',history)
    write(args.out/'summary.json',{'contract':contract,'step':step,'history':history,'checkpoint_sha256':sha(resume),
          'status':'complete' if step==args.updates else 'chunk_complete_fit_remaining'})
    print(json.dumps(event),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--objective',choices=['auf','ce'],required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--validation-data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--records',type=int,required=True)
    parser.add_argument('--updates',type=int,default=1024)
    parser.add_argument('--chunk-updates',type=int,default=128)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--lr',type=float,required=True)
    main(parser.parse_args())
