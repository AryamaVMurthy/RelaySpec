"""Record historical tensor comparison without confusing new-data training with bitwise reproduction."""
import os,sys,json,subprocess
from pathlib import Path
import torch
root=Path('/scratch/aryama.murthy/transfer-reproduction-20260907');p=root/'full-package';w=root/'work'
r=subprocess.run([sys.executable,str(p/'src/check_mapper.py')],check=False)
f=w/'validation/mapper_reproduction.json'
assert f.exists(),'Historical checker failed without producing its report'
a=json.loads(f.read_text());assert r.returncode==(0 if a['exact'] else 1)
assert a['actual'].keys()==a['expected'].keys()
for k,v in a['actual'].items():
 assert v['shape']==a['expected'][k]['shape'] and v['dtype']==a['expected'][k]['dtype']
 if k in ['fusion','norm']:assert v==a['expected'][k],('Frozen source changed',k)
q=torch.load(w/'fit/final.pt',map_location='cpu',weights_only=True)
assert all(torch.isfinite(v).all() for v in q['model'].values())
s=json.loads((w/'fit/summary.json').read_text());contract=json.loads((w/'cache/setup/data_contract.json').read_text())
assert s['actual_train_examples']==16384 and s['selected_epoch']==2 and len(s['history'])==3
assert s['positions_per_epoch']==contract['totals']['train']['sampled_positions']
assert s['parameters']==52428800 and s['actual_dev_examples']==0
assert [x['epoch'] for x in s['history']]==[0,1,2]
a['continuation_basis']='Fresh-data recipe reproduction explicitly accepted; frozen tensors, parameter shapes, finite values, complete data coverage and three fixed epochs verified. Exact historical mapper match remains a separate recorded outcome.'
(w/'validation/full_mapper_audit.json').write_text(json.dumps(a,indent=2))
print(json.dumps({'historical_tensors_exact':a['exact'],'training_contract_verified':True}))
