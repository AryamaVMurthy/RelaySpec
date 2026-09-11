from common import *
import torch
train=json.loads((R/'setup/train.json').read_text());test=json.loads((R/'setup/eval.json').read_text())
assert len(train)==4096 and len(test)==128
assert len({x['group_id'] for x in train})==4096
assert not {x['group_id'] for x in train}&{x['group_id'] for x in test}
paths=sorted((R/'features/full').glob('*.pt'));assert len(paths)==128
ids=[]
for p in paths:
 z=torch.load(p,weights_only=True,mmap=True);assert len(z['rows'])==len(z['features'])==32
 for row,h in zip(z['rows'],z['features']):
  assert h.shape==(len(row['full_ids']),12800) and h.dtype==torch.bfloat16
  assert torch.isfinite(h).all();assert row['full_ids']==row['prompt_token_ids']+row['output_ids'];assert len(row['output_ids'])<=4096
  ids.append(row['group_id'])
assert len(set(ids))==4096 and set(ids)=={x['group_id'] for x in train}
print('4096 disjoint examples / 128 dense shards validated. Loss still samples 8 anchors/example.')
