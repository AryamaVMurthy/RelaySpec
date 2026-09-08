"""Aggregate the frozen 128-request family check, retaining every disagreement."""
import argparse,json,hashlib
from pathlib import Path
from collections import defaultdict
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--report',required=True);a=p.parse_args()
root=Path(a.root);report=Path(a.report)
expected={r['problem_id'] for r in json.loads((report/'manifest.json').read_text())['records']}
groups=defaultdict(dict);completed=[]
for wave in range(16):
 for lane in range(4):
  run=root/f'wave{wave}'/f'lane{lane}'/'shared'
  raw=run/'benchmark-rank0.jsonl'
  if not raw.exists():continue
  if (run.parent/'COMPLETE').exists():completed.append([wave,lane])
  family='llama' if lane<2 else 'cross'
  for line in raw.read_text().splitlines():
   try:r=json.loads(line)
   except json.JSONDecodeError:continue # a live writer can have a partial last row
   name=r['method']; method='ar' if name=='native_ar' else ('selected' if name=='relay_candidate_1' else 'old')
   key=(r['problem_id'],r.get('turn_index',0),r.get('repetition',0))
   assert key[0] in expected,(run,key)
   assert key not in groups[family,method],('duplicate',family,method,key)
   ids=r['output_token_ids']
   import struct
   assert hashlib.sha256(struct.pack('<'+'i'*len(ids),*ids)).hexdigest()==r['output_hash']
   assert len(ids)==r['output_tokens']
   groups[family,method][key]=r
summary={'complete_lanes':len(completed),'expected_lanes':64,'max_new_tokens':2048,'families':{},'complete':len(completed)==64}
for family in ['llama','cross']:
 ar=groups[family,'ar']; methods={}
 for method in ['ar','old','selected']:
  rows=groups[family,method]; paired=set(rows)&set(ar); mismatches=[];matches=0
  for key in sorted(paired):
   actual=rows[key]['output_token_ids'];ref=ar[key]['output_token_ids']
   if actual==ref:matches+=1
   else:
    i=next((i for i,(x,y) in enumerate(zip(actual,ref)) if x!=y),min(len(actual),len(ref)))
    mismatches.append({'problem_id':key[0],'first_divergence_token_0based':i,'ar_token':ref[i] if i<len(ref) else None,'method_token':actual[i] if i<len(actual) else None,'ar_length':len(ref),'method_length':len(actual)})
  tokens=sum(r['output_tokens'] for r in rows.values());decode=sum(r['decode_seconds'] for r in rows.values());request=sum(r['request_seconds'] for r in rows.values())
  methods[method]={'requests':len(rows),'paired_requests':len(paired),'exact_matches':matches,'mismatches':mismatches,'output_tokens':tokens,'decode_tps':tokens/decode if decode else None,'request_tps':tokens/request if request else None,'cap_reached':sum(r['output_tokens']>=2048 for r in rows.values())}
  if summary['complete']:assert {k[0] for k in rows}==expected and len(rows)==128
 summary['families'][family]=methods
(report/'status.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
