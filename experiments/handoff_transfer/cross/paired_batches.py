"""Stream paired positions while preserving equal total mass per training record."""
import json
import random
from pathlib import Path
import torch
from experiments.handoff_transfer.matrix.prepare import digest


def paired_index(manifest):
    data=json.loads(Path(manifest).read_text());items=[]
    for entry in data['index']:
        path=Path(entry['chunk'])/f"paired/{entry['row_index']:05d}.pt"
        meta=json.loads(path.with_suffix('.json').read_text())
        assert meta['group_id']==entry['group_id'] and meta['manifest_sha256']==entry['rollout_sha256']
        assert meta['positions']>0 and path.is_file()
        items.append(dict(path=str(path),group_id=entry['group_id'],positions=meta['positions'],sha256=meta['sha256']))
    assert len(items)==data['records'] and len({x['group_id'] for x in items})==len(items)
    return items


def batches(items, batch_size=2048, seed=42):
    if batch_size<1:raise ValueError('Positive position batch size required')
    rng=random.Random(seed);order=list(range(len(items)));rng.shuffle(order)
    xs=[];ys=[];ws=[];pending=0
    for index in order:
        item=items[index];path=Path(item['path'])
        assert digest(path)==item['sha256'],'Paired feature file changed'
        record=torch.load(path,weights_only=True,mmap=True)
        count=item['positions']
        assert record['group_id']==item['group_id']
        assert record['x'].shape==(count,20480) and record['y'].shape==(count,12800)
        positions=list(range(count));rng.shuffle(positions)
        x=record['x'][positions];y=record['y'][positions]
        start=0
        while start<count:
            take=min(batch_size-pending,count-start)
            xs.append(x[start:start+take]);ys.append(y[start:start+take])
            ws.append(torch.full((take,),1./count,dtype=torch.float32))
            start+=take;pending+=take
            if pending==batch_size:
                yield torch.cat(xs),torch.cat(ys),torch.cat(ws)
                xs=[];ys=[];ws=[];pending=0
    if pending:yield torch.cat(xs),torch.cat(ys),torch.cat(ws)
