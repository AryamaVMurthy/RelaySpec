"""Bounded-memory, record-weighted fitting on shared raw paired features."""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from train_zip_family_pilot import Context
from relayspec.relay import TargetFeatureRelay


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pair', choices=['llama', 'cross'], required=True)
    p.add_argument('--objective', choices=['zip', 'dense'], required=True)
    p.add_argument('--records', type=int, required=True)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(os.environ['FAMILY_SCALE_CACHE']) / a.pair
    meta = torch.load(root/'metadata.pt', weights_only=True)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    (out/'trial.json').write_text(json.dumps(vars(a), indent=2))
    teacher = Context(meta['fusion'], meta['norm'], d8=meta['di'], d4=meta['do']).cuda()
    if a.objective == 'zip':
        model = teacher
    else:
        teacher.requires_grad_(False)
        model = TargetFeatureRelay(target_hidden_size=meta['di'], num_taps=5,
            draft_hidden_size=meta['do'], eps=1e-6).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0, fused=True)

    def batch(split, indices):
        records = [torch.load(root/split/f'{i:05d}.pt', weights_only=True,
                              mmap=True) for i in indices]
        assert all(r['row'] == i and r['split'] == split for r,i in zip(records,indices))
        x = torch.cat([r['x'] for r in records]).cuda()
        y = torch.cat([r['y'] for r in records]).cuda()
        w = torch.cat([torch.full((len(r['x']),),1/len(r['x'])) for r in records]).cuda()
        return x,y,w

    def losses(x,y):
        if a.objective == 'zip':
            return model.loss(x,y)
        target = teacher.frozen_norm(torch.nn.functional.linear(y,teacher.fusion))
        pred = teacher.frozen_norm(model(x.unsqueeze(0)).squeeze(0))
        return (pred.float()-target.float()).square().sum(-1)/(target.float().square().sum(-1)+1e-6)

    @torch.no_grad()
    def validation():
        total = 0.
        for start in range(0,256,64):
            x,y,w = batch('validation',list(range(start,start+64)))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                total += float((losses(x,y)*w).sum())
        return total/256

    saves = sorted(set([e for e in [1,3,6,12] if e <= a.epochs]+[a.epochs]))
    steps = math.ceil(a.records/64)*a.epochs
    warm = max(1,int(.05*steps)); step = 0; positions_seen = 0
    start = time.perf_counter()
    print(json.dumps({'initial_validation':validation()}), flush=True)
    for epoch in range(1,a.epochs+1):
        order = torch.randperm(a.records,generator=torch.Generator().manual_seed(42+epoch)).tolist()
        total = 0.; epoch_start = time.perf_counter()
        for j in range(0,len(order),64):
            ids = order[j:j+64]; x,y,w = batch('train',ids)
            positions_seen += len(x)
            rate = (step+1)/warm if step < warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,steps-warm)))
            opt.param_groups[0]['lr'] = a.lr*rate
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = (losses(x,y)*w).sum()/len(ids)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step(); step += 1
            total += float(loss.detach())*len(ids)
        event = {'epoch':epoch,'updates':step,'positions_seen':positions_seen,'train_loss':total/a.records,
                 'validation_loss':validation(),'seconds':time.perf_counter()-epoch_start}
        if epoch in saves:
            if a.objective == 'zip':
                weight = model.folded().detach()
                with torch.no_grad():
                    sample = x[:128].float()
                    ref,_ = model(sample)
                    folded = model.frozen_norm(torch.nn.functional.linear(sample,weight))
                    err = float((ref-folded).square().sum(-1).mean()/ref.square().sum(-1).mean())
                    assert err < 1e-8, err
                    event['fp32_fold_relative_mse'] = err
            else:
                weight = model.projection.weight.detach()
            torch.save({'relay':{'projection.weight':weight.cpu()},
                'relay_architecture':'raw_linear' if a.objective=='zip' else 'normalized_linear',
                'target_layer_ids':meta['target_taps'], 'source_layer_ids':meta['source_taps'],
                'proposer_family':'dflash','feature_objective':a.objective,
                'steps':step,'epoch':epoch,'training_records':a.records,
                'positions_seen':positions_seen,
                'train_manifest_sha256':meta['train_sha256']},out/f'epoch-{epoch}.pt')
        with (out/'history.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
    (out/'COMPLETE.json').write_text(json.dumps({'seconds':time.perf_counter()-start,
        'records':a.records,'epochs':a.epochs,'steps':step,'positions_seen':positions_seen,'job_id':os.environ['SLURM_JOB_ID']}))


if __name__ == '__main__':
    main()
