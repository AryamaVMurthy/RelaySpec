import torch,json,hashlib
from paths import WORK,put
files=sorted((WORK/'cache/features/train/8').glob('*.pt'));assert len(files)==128
count=positions=0
for p in files:
 a=torch.load(p,weights_only=True);b=torch.load(WORK/'cache/features/train/4'/p.name,weights_only=True)
 assert a['group_ids']==b['group_ids'] and a['rollout_sha256']==b['rollout_sha256']
 for x,y,i,j in zip(a['features'],b['features'],a['positions'],b['positions']):
  assert torch.equal(i,j) and x.shape==(len(i),20480) and y.shape==(len(i),12800)
  assert torch.isfinite(x).all() and torch.isfinite(y).all();count+=1;positions+=len(i)
assert count==16384
put(WORK/'cache/manifest.json',{'complete':True,'examples':count,'positions':positions,'layers':[1,9,17,25,33]})
