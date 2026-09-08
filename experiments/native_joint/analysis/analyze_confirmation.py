"""Audit full coverage before reporting frozen confirmation estimates."""
import json
from pathlib import Path
import math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
freeze=json.loads((ROOT/'data/frozen-native-confirmation.json').read_text())
campaign=json.loads((ROOT/'data/confirmation-campaign.json').read_text())
manifest=json.loads((ROOT/'data/confirmation.json').read_text())['records']
records={r['problem_id']:r for r in manifest}
jobs=json.loads((ROOT/'reports/confirmation-jobs.json').read_text())
methods=['native','compact_linear','candidate','reference']
rows={};ar={};complete=[];pending=[]
for job in jobs:
    directory=ROOT/f"reports/run-{job['job']}"
    for lane in range(4):
        directory_lane=directory/f'lane{lane}'
        status_path=directory/f'lane{lane}-status.json'
        if not status_path.exists():pending.append([job['job'],lane]);continue
        status=json.loads(status_path.read_text())
        if status['exit_code']!=0:raise RuntimeError(f'Failed confirmation lane: {job}, {lane}, {status}')
        config=json.loads((directory/f'lane{lane}.json').read_text())
        protocol=json.loads((directory_lane/'evaluation-protocol.json').read_text())
        provenance=json.loads((directory_lane/'provenance.json').read_text())
        assert config['phase']=='confirmation'
        assert protocol['frozen_selection_sha256']==campaign['freeze_sha256']
        assert protocol['protocol']==freeze['protocol']
        assert provenance['gpu']=='NVIDIA L40S'
        assert {k:provenance[k] for k in ['torch','transformers']}==freeze['protocol']['required_runtime']
        assert provenance['target']==freeze['protocol']['target'] and provenance['draft']==freeze['protocol']['draft']
        assert provenance['source_commit']==freeze['protocol']['source_commit']
        expected=[r['problem_id'] for b in config['benchmarks'] for r in [r for r in manifest if r['benchmark']==b][config['eval_offset']:config['eval_offset']+config['requests_per_benchmark']]]
        assert len(expected)==2 and len(set(expected))==2
        result_dir=directory_lane/config['variants'][0]['name']
        result=json.loads((result_dir/'result.json').read_text());assert result['status']=='pass'
        local=[json.loads(line) for line in (result_dir/'evaluation.jsonl').read_text().splitlines()]
        assert len(local)==2*2*len(methods)
        assert {(r['problem_id'],r['method'],r['repeat']) for r in local}=={(p,m,t) for p in expected for m in methods for t in range(2)}
        for r in local:
            assert r['output_tokens']==len(r['tokens'])<=2048
            assert math.isfinite(r['seconds']) and r['seconds']>0
            assert r['benchmark']==records[r['problem_id']]['benchmark']
            key=(r['method'],r['problem_id'],r['repeat'])
            assert key not in rows;rows[key]=r
        marker=json.loads((directory_lane/'ar-quality-complete.json').read_text())
        assert marker=={'status':'pass','requests':2,'output_cap':2048}
        local_ar=[json.loads(line) for line in (directory_lane/'ar-quality.jsonl').read_text().splitlines()]
        assert len(local_ar)==2 and {r['problem_id'] for r in local_ar}==set(expected)
        for r in local_ar:
            assert r['problem_id'] not in ar and r['output_tokens']==len(r['tokens'])<=2048
            ar[r['problem_id']]=r
        complete.append([job['job'],lane])
progress={'completed_lanes':len(complete),'total_lanes':64,'completed_requests':len(ar),'total_requests':128,'pending':pending,'status':'complete' if len(complete)==64 else 'running'}
(ROOT/'reports/confirmation-progress.json').write_text(json.dumps(progress,indent=2)+'\n')
print(json.dumps(progress))
if pending:raise SystemExit(0)
assert set(ar)==set(records)==set(freeze['problem_ids']) and len(rows)==128*4*2
for m in methods:
    for p in records:
        assert all(rows[m,p,0][k]==rows[m,p,1][k] for k in ['tokens','acceptance_lengths'])
ordered=sorted(records)
arrays={m:np.array([[sum(rows[m,p,t]['output_tokens'] for t in range(2)),sum(rows[m,p,t]['seconds'] for t in range(2))] for p in ordered]) for m in methods}
rng=np.random.default_rng(1729)
# Preserve the32/32/32/32 workload mix while resampling whole paired questions.
indices=np.concatenate([rng.choice([i for i,p in enumerate(ordered) if records[p]['benchmark']==b],size=(10000,32),replace=True) for b in ['gsm8k','math500','humaneval','mtbench']],axis=1)
summary=[]
for b in ['all','gsm8k','math500','humaneval','mtbench']:
    ids=[p for p in ordered if b=='all' or records[p]['benchmark']==b]
    selected=[ordered.index(p) for p in ids]
    draws=indices if b=='all' else rng.choice(selected,size=(10000,len(ids)),replace=True)
    samples={m:arrays[m][draws].sum(1) for m in methods}
    for m in methods:
        data=arrays[m][selected].sum(0);tps=data[0]/data[1]
        ratios={}
        for reference in ['native','reference']:
            point=arrays[reference][selected].sum(0)
            a,z=samples[m],samples[reference]
            ratios[reference]={'ratio':tps/(point[0]/point[1]),'ci95':np.quantile((a[:,0]/a[:,1])/(z[:,0]/z[:,1]),[.025,.975]).tolist()}
        summary.append({'benchmark':b,'method':m,'requests':len(ids),'tps':tps,'comparisons':ratios,'exact_native':sum(rows[m,p,0]['tokens']==rows['native',p,0]['tokens'] for p in ids),'exact_ar':sum(rows[m,p,0]['tokens']==ar[p]['tokens'] for p in ids),'capped':sum(rows[m,p,0]['capped'] for p in ids),'mean_progress':sum(sum(rows[m,p,t]['acceptance_lengths']) for p in ids for t in range(2))/sum(len(rows[m,p,t]['acceptance_lengths']) for p in ids for t in range(2)),'output_tokens_per_round':float(data[0])/sum(len(rows[m,p,t]['acceptance_lengths']) for p in ids for t in range(2)),'amortized_seconds_per_round':float(data[1])/sum(len(rows[m,p,t]['acceptance_lengths']) for p in ids for t in range(2))})
(ROOT/'reports/confirmation-summary.json').write_text(json.dumps({'status':'complete','coverage':progress,'rows':summary,'freeze_sha256':campaign['freeze_sha256'],'interval_scope':'Stratified paired question bootstrap; timing repeats clustered. Excludes fitting-seed uncertainty.'},indent=2)+'\n')
with (ROOT/'reports/confirmation-unique-outputs.jsonl').open('w') as f:
    for p in ordered:
        for m in methods:f.write(json.dumps(rows[m,p,0])+'\n')
        f.write(json.dumps(ar[p])+'\n')
lines=['# Frozen128-request confirmation','','All128 requests completed:32 each GSM8K, MATH500, HumanEval and MTBench, output cap2048, two timing repeats. Full wall time includes prefill. AR is a single-generation quality reference outside repeated timing. No configuration changed from the frozen selection.','','| Method | TPS | / Original [95% interval] | / Full DDTree [95% interval] | Exact original | Exact AR | Capped |','|---|---:|---|---|---:|---:|---:|']
for r in summary:
    if r['benchmark']!='all':continue
    a,z=r['comparisons']['native'],r['comparisons']['reference']
    lines.append(f"| {r['method']} | {r['tps']:.1f} | {a['ratio']:.3f} [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}] | {z['ratio']:.3f} [{z['ci95'][0]:.3f}, {z['ci95'][1]:.3f}] | {r['exact_native']}/128 | {r['exact_ar']}/128 | {r['capped']}/128 |")
lines+=['','`compact_linear` isolates joint training; `candidate` is that checkpoint with DDTree47; `reference` is the released full drafter with DDTree63. DDTree is existing prior art. Output identity is measured, not assumed; see the separate quality audit before making accuracy claims. Intervals resample paired questions within workload, not independent timing repeats; they exclude training-seed variation.']
(ROOT/'reports/CONFIRMATION.md').write_text('\n'.join(lines)+'\n')
