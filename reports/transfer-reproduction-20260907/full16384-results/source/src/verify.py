import argparse,json,hashlib
from paths import WORK,PACKAGE,put
p=argparse.ArgumentParser();p.add_argument('--tag',default='final');p.add_argument('--count',type=int,default=128);a=p.parse_args()
groups={};sources={}
for side in ['ar8','native8','mapped']:
 files=sorted((WORK/'measurements'/a.tag).glob('worker_*/'+side+'-r0.jsonl'))
 rows=[json.loads(l) for path in files for l in path.read_text().splitlines()];assert len(rows)==a.count
 assert all(x['timing_valid'] and x['output_tokens']==len(x['output_ids']) for x in rows)
 by={x['group_id']:x for x in rows};assert len(by)==a.count
 groups[side]=by;sources[side]=[{ 'path':str(x),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in files]
keys=groups['ar8'].keys();assert all(x.keys()==keys for x in groups.values())
differences=[];reference_differences=[]
expected={x['group_id']:x for x in map(json.loads,(PACKAGE/'reference/ar8.jsonl').read_text().splitlines())}
for k in keys:
 x=[rows[k] for rows in groups.values()]
 assert x[0]['prompt_ids']==x[1]['prompt_ids']==x[2]['prompt_ids']==expected[k]['prompt_ids']
 if not (x[0]['output_ids']==x[1]['output_ids']==x[2]['output_ids'] and x[0]['finish_reason']==x[1]['finish_reason']==x[2]['finish_reason']):differences.append(k)
 wanted=expected[k]['output_ids'] if a.count==128 else expected[k]['output_ids'][:256]
 if any(t['output_ids']!=wanted for t in x):reference_differences.append(k)
stats={side:{'tokens':sum(len(x['output_ids']) for x in rows.values()),'seconds':sum(x['wall_seconds'] for x in rows.values())} for side,rows in groups.items()}
for side,x in stats.items():x['tokens_per_second']=x['tokens']/x['seconds'];x['speedup_vs_ar8']=stats['ar8']['seconds']/x['seconds']
result={'all_equal':not differences,'count':a.count,'different_groups':differences,'different_from_historical_output':reference_differences,'stats':stats,'mapper_vs_native':stats['native8']['seconds']/stats['mapped']['seconds'],'sources':sources}
put(WORK/'results'/(a.tag+'.json'),result);print(json.dumps(result,indent=2))
assert not differences,'Pipelines disagree: preserve raw outputs and investigate'
assert not reference_differences,'Pipelines agree with each other but differ from historical reference'
