"""Separate adaptive pilots from added-request tree measurements."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('reports')
manifest=json.loads(Path('data/screening.json').read_text())['records']
pilot={r['problem_id'] for b in ['gsm8k','math500','humaneval','mtbench'] for r in [r for r in manifest if r['benchmark']==b][:2]}
rows=[]
for path in sorted(root.glob('run-*/lane[0-3]/*/result.json')):
    lane_config=path.parents[2]/(path.parents[1].name+'.json')
    if lane_config.exists() and json.loads(lane_config.read_text()).get('phase')=='confirmation':
        continue
    result=json.loads(path.read_text())
    if result.get('variant',{}).get('kind') not in ['tree_branches','ddtree'] or result.get('status')!='pass': continue
    raw=[json.loads(line) for line in (path.parent/'evaluation.jsonl').read_text().splitlines()]
    for subset in ['all','additional']:
        data=[r for r in raw if subset=='all' or r['problem_id'] not in pilot]
        ids=sorted({r['problem_id'] for r in data})
        if not ids: continue
        arr=[]
        for method in ['native','candidate']:
            arr.append(np.array([[sum(r['output_tokens'] for r in data if r['problem_id']==i and r['method']==method),sum(r['seconds'] for r in data if r['problem_id']==i and r['method']==method)] for i in ids]))
        base,arm=arr
        idx=np.random.default_rng(1729).integers(len(ids),size=(10000,len(ids)))
        a,b=arm[idx].sum(1),base[idx].sum(1)
        ratios=(a[:,0]/a[:,1])/(b[:,0]/b[:,1])
        tps=[x[:,0].sum()/x[:,1].sum() for x in arr]
        rows.append(dict(run=path.parents[2].name,name=result['variant']['name'],subset=subset,requests=len(ids),ratio=tps[1]/tps[0],ci95=np.quantile(ratios,[.025,.975]).tolist(),native_tps=tps[0],candidate_tps=tps[1],path=str(path)))
(root/'tree-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
selected=[r for r in rows if r['requests']>=24]
lines=['# Packed tree development results','','These are adaptive development results. Additional excludes the initial eight pilot requests, but is not the reserved confirmation set. Request bootstrap intervals exclude search selection uncertainty and do not establish a10% lower bound. Timings include prefill; cap512 unless the variant explicitly states otherwise. See raw results for numerical output agreement.','','| Configuration | Requests | Candidate / native TPS | Ratio [paired95% interval] |','|---|---:|---:|---|']
for r in selected:
    lines.append(f"| {r['run']} {r['name']} ({r['subset']}) | {r['requests']} | {r['candidate_tps']:.1f} / {r['native_tps']:.1f} | {r['ratio']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] |")
lines+=['','Direct prior art: DDTree (Ringel and Romano, arXiv2604.12989). Our baseline adapts its pinned MIT heap builder into the shared runtime; no general tree novelty claim.','','Selected four BF16 divergent cases matched exactly after FP32 parameter conversion of both models. All had packed-target top-logit ties. This is diagnostic evidence for numerical sensitivity in those cases, not universal equality or task-quality validation.']
(root/'TREE_RESULTS.md').write_text('\n'.join(lines)+'\n')
if selected:
    fig,ax=plt.subplots(figsize=(9,max(3,len(selected)*.43)),layout='constrained')
    for i,r in enumerate(selected):
        ax.errorbar(r['ratio'],i,xerr=[[r['ratio']-r['ci95'][0]],[r['ci95'][1]-r['ratio']]],fmt='o',color='tab:blue' if r['subset']=='all' else 'tab:orange')
    ax.set_yticks(range(len(selected)),[r['name']+' / '+str(r['requests'])+' requests' for r in selected]);ax.axvline(1,color='black',lw=1);ax.axvline(1.1,color='green',ls='--');ax.set_xlabel('End-to-end throughput / original DFlash');ax.grid(axis='x',alpha=.2)
    fig.savefig(root/'tree-breadth.png',dpi=180);fig.savefig(root/'tree-breadth.pdf')
print(json.dumps(selected,indent=2))
