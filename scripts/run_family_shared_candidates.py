"""Share frozen models and AR across candidate maps; retain individual views."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

p=argparse.ArgumentParser()
p.add_argument('--list',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
paths=json.loads(Path(a.list).read_text())
configs=[yaml.safe_load(Path(path).read_text()) for path in paths]
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
cfg=json.loads(json.dumps(configs[0]))
base_method=next(m for m in cfg['benchmark']['methods'] if m!='native_ar')
methods=[base_method]+[f'relay_candidate_{i}' for i in range(1,len(configs))]
for c in configs:
    for key in ['target','source_trunk','proposer','generation']:
        assert c[key]==cfg[key],key
    for key in ['manifest_path','max_prompts','precision','target_head_precision','warmups']:
        assert c['benchmark'][key]==cfg['benchmark'][key],key
    assert c['benchmark'].get('drafter_precision','bfloat16') == cfg['benchmark'].get('drafter_precision','bfloat16')
cfg['benchmark']['methods']=['native_ar',*methods]
cfg['relay_probe']['variants']={name:c['relay_probe']['checkpoint_path'] for name,c in zip(methods[1:],configs[1:])}
cfg['relay_probe']['variant_block_sizes']={name:c['benchmark']['block_size'] for name,c in zip(methods[1:],configs[1:])}
cfg['relay_probe']['cross_family_variants']=base_method=='relay_p_cross_family'
shared=out/'shared';shared.mkdir(exist_ok=False)
combined=shared/'config.yaml';combined.write_text(yaml.safe_dump(cfg,sort_keys=False))
with (shared/'stdout.log').open('w') as stdout,(shared/'stderr.log').open('w') as stderr:
    subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',
        'scripts/benchmark_mixed_precision_relay.py','--config',str(combined)],
        env=dict(os.environ,RELAYSPEC_OUTPUT=str(shared.resolve())),stdout=stdout,stderr=stderr,check=True)
summary=json.loads((shared/'benchmark-summary.json').read_text())
raw=[json.loads(s) for s in (shared/'benchmark-rank0.jsonl').read_text().splitlines()]
results=[]
for path,method in zip(paths,methods):
    run=out/Path(path).stem;run.mkdir(exist_ok=False)
    pair={'native_ar':summary['methods']['native_ar'],base_method:summary['methods'][method]}
    data={'paired_requests':summary['paired_requests'],'methods':pair}
    (run/'benchmark-summary.json').write_text(json.dumps(data,indent=2))
    rows=[]
    for row in raw:
        if row.get('method') not in ['native_ar',method]:
            continue
        r=dict(row)
        if r['method']==method:
            r['method']=base_method
        rows.append(r)
    (run/'benchmark-rank0.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    valid=all(r['exact_sequence_matches']==r['requests']>0 for r in pair.values())
    results.append({'config':path,'output':str(run),'exact_gate':valid,
        'shared_campaign':str(shared),'resident_maps':len(configs),**data})
(out/'results.json').write_text(json.dumps(results,indent=2))
(out/'COMPLETE').write_text('Shared-model screen complete; inspect exact_gate before promotion.\n')
print(json.dumps(results),flush=True)
