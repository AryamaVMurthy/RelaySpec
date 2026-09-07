import torch,json,hashlib
from paths import WORK,PACKAGE,put
q=torch.load(WORK/'fit/final.pt',map_location='cpu',weights_only=True)
actual={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()) for k,v in q['model'].items()}
expected=json.loads((PACKAGE/'provenance/mapper_tensor_hashes.json').read_text())
put(WORK/'validation/mapper_reproduction.json',{'exact':actual==expected,'actual':actual,'expected':expected})
assert actual==expected,'Mapper differs from historical tensor hashes; inspect training-data/features/numerics before claiming bitwise reproduction'
print('All mapper tensors match historical hashes')
