"""Tensorize cross-family records without treating target IDs as source IDs."""
import torch

def collate(records,block_size=16):
    if not records:raise ValueError('Empty batch')
    length=max(len(r['target_ids']) for r in records)
    width=records[0]['features'].shape[-1]
    hidden=torch.zeros(len(records),length,width,dtype=torch.bfloat16)
    anchors=torch.zeros(len(records),length,dtype=torch.long)
    eligible=torch.zeros(len(records),length)
    labels=torch.zeros(len(records),length,block_size,dtype=torch.long)
    validity=torch.zeros_like(labels,dtype=torch.bool)
    for i,r in enumerate(records):
        n=len(r['target_ids'])
        if r['features'].shape!=(n,width):raise ValueError('Target feature/token grid mismatch')
        hidden[i,:n]=r['features']
        seen=set()
        for block in r['alignment']['blocks']:
            k=block['target_anchor'];values=block['source_labels']
            if k in seen or not 0<k<n or block['context_exclusive_end']!=k:
                raise ValueError('Invalid causal anchor')
            if not 2<=len(values)<=block_size:raise ValueError('Invalid source block')
            seen.add(k)
            anchors[i,k]=values[0];eligible[i,k]=1
            labels[i,k,:len(values)]=torch.tensor(values)
            validity[i,k,:len(values)]=True
        if not seen:raise ValueError('Record has no aligned anchors')
    return anchors,hidden,eligible,labels,validity
