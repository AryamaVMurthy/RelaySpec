import json,hashlib,argparse
from paths import WORK,PACKAGE,put
p=argparse.ArgumentParser();p.add_argument('--accept-new-data',action='store_true',help='Explicitly record a new training realization; not a bitwise historical reproduction');a=p.parse_args()
expected=json.loads((PACKAGE/'provenance/rollout_shards.json').read_text());files=sorted((WORK/'common/rollouts/train').glob('*.jsonl'));assert len(files)==128
seen=set();mismatch=[];tokens=0
for path in files:
 rows=[json.loads(l) for l in path.read_text().splitlines()];assert len(rows)==128
 for x in rows:
  assert x['group_id'] not in seen;seen.add(x['group_id']);assert x['full_ids']==x['prompt_token_ids']+x['output_ids'];assert 0<len(x['output_ids'])<=4096;tokens+=len(x['output_ids'])
 if hashlib.sha256(path.read_bytes()).hexdigest()!=expected[path.stem+'.meta.json']['sha256']:mismatch.append(path.name)
put(WORK/'validation/rollout_reproduction.json',{'examples':len(seen),'tokens':tokens,'different_historical_shards':mismatch,'explicit_new_data_acceptance':a.accept_new_data})
assert not mismatch or a.accept_new_data, 'Training rollouts differ; inspect provenance/README before explicitly accepting a new realization'
