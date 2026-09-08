"""Report compact candidates against their simultaneous full DDTree control."""
import json
from pathlib import Path
import numpy as np
root=Path('reports')
rows=[]
for path in sorted(root.glob('run-*/lane[0-3]/*/result.json')):
    lane_config=path.parents[2]/(path.parents[1].name+'.json')
    if lane_config.exists() and json.loads(lane_config.read_text()).get('phase')=='confirmation':
        continue
    result=json.loads(path.read_text())
    if result.get('status')!='pass' or 'reference_ratio' not in result:continue
    raw=[json.loads(line) for line in (path.parent/'evaluation.jsonl').read_text().splitlines()]
    ids=sorted({r['problem_id'] for r in raw})
    arrays={name:np.asarray([[sum(r['output_tokens'] for r in raw if r['method']==name and r['problem_id']==i),sum(r['seconds'] for r in raw if r['method']==name and r['problem_id']==i)] for i in ids]) for name in ['native','candidate','reference']}
    draws=np.random.default_rng(1729).integers(len(ids),size=(10000,len(ids)))
    arm,base=arrays['candidate'][draws].sum(1),arrays['reference'][draws].sum(1)
    ci=np.quantile((arm[:,0]/arm[:,1])/(base[:,0]/base[:,1]),[.025,.975]).tolist()
    rows.append(dict(name=result['variant']['name'],run=path.parents[2].name,requests=len(ids),native_ratio=result['native_ratio'],reference_ratio=result['reference_ratio'],reference_ci95=ci,summary=result['summary'],path=str(path)))
(root/'tree-reference-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
lines=['# Direct comparison with full DDTree','','All three arms run in the same GPU process, with rotated order, on the same requests. These are adaptive development tests. The adapted DDTree algorithm uses the pinned released draft and target; budgets exclude its bonus root. Candidate training is additional cost, recorded in its checkpoint provenance. Paired request intervals cluster timing repeats and exclude selection uncertainty.','','| Candidate | Original DFlash TPS | Full DDTree TPS | Candidate TPS | / Original | / Full DDTree [95% interval] |','|---|---:|---:|---:|---:|---|']
for r in rows:
    s=r['summary'];lines.append(f"| {r['name']} | {s['native']['tps']:.1f} | {s['reference']['tps']:.1f} | {s['candidate']['tps']:.1f} | {r['native_ratio']:.3f} | {r['reference_ratio']:.3f} [{r['reference_ci95'][0]:.3f}, {r['reference_ci95'][1]:.3f}] |")
lines+=['','The compact two-tap model beats original single-path DFlash but loses to full DDTree. This comparison does not support an added benefit from joint compression.']
(root/'TREE_REFERENCES.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(rows,indent=2))
