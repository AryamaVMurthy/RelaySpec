"""Summarize observed outcomes without treating selected screens as confirmation."""
import argparse
import json
import random
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--root',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
root=Path(a.root); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
fits=[]; screens=[]; blocks=[]; final=[]
for path in sorted((root/'fits').glob('n*/history.jsonl')):
    trial=json.loads((path.parent/'trial.json').read_text())
    fits.append({'trial':trial,'history':[json.loads(s) for s in path.read_text().splitlines()],
                 'complete':(path.parent/'COMPLETE.json').exists()})
for stage,dest in [('screen-*',screens),('blocks',blocks),('final',final)]:
    for path in sorted(root.glob(stage+'/lane*/results.json')):
        dest.extend(json.loads(path.read_text()))
comparison=[]
if len(final)==4:
    lanes={}
    for r in final:
        lane=int(Path(r['output']).parent.name.removeprefix('lane'))
        raw=[json.loads(s) for s in (Path(r['output'])/'benchmark-rank0.jsonl').read_text().splitlines()]
        method=next(k for k in r['methods'] if k!='native_ar')
        rows={ (x['problem_id'],x['turn_index'],x['repetition']):x for x in raw if x.get('method')==method }
        assert r['exact_gate'] and len(rows)==16
        lanes[lane]=(r,rows,method)
    for pair,old_lane,new_lane in [('llama',0,1),('cross',2,3)]:
        old,old_rows,old_method=lanes[old_lane]; new,new_rows,new_method=lanes[new_lane]
        assert old_rows.keys()==new_rows.keys()
        keys=sorted(old_rows)
        assert all(old_rows[k]['output_hash']==new_rows[k]['output_hash'] and
                   old_rows[k]['output_tokens']==new_rows[k]['output_tokens'] for k in keys)
        def ratio(selected,field):
            return sum(old_rows[k][field] for k in selected)/sum(new_rows[k][field] for k in selected)
        rng=random.Random(1729)
        sampled=[ratio(rng.choices(keys,k=len(keys)),'request_seconds') for _ in range(10000)]
        sampled.sort()
        def request_tps(rows):
            return sum(x['output_tokens'] for x in rows.values())/sum(x['request_seconds'] for x in rows.values())
        comparison.append({'pair':pair,'requests':16,'exact_same_outputs':True,
            'old_decode_tps':old['methods'][old_method]['decode_tokens_per_second'],
            'selected_decode_tps':new['methods'][new_method]['decode_tokens_per_second'],
            'old_request_tps':request_tps(old_rows),'selected_request_tps':request_tps(new_rows),
            'request_speedup_vs_old':ratio(keys,'request_seconds'),
            'paired_request_bootstrap_95ci':[sampled[250],sampled[9750]],
            'ci_scope':'Request resampling only; excludes run-to-run and fitting-seed uncertainty',
            'old_config':old['config'],'selected_config':new['config']})
data={'fits':fits,'screening':screens,'block_tuning':blocks,'final':final,'comparison':comparison}
(out/'results.json').write_text(json.dumps(data,indent=2))
lines=['# Expanded family-transfer results','',
    f'Completed fitting runs: {sum(f["complete"] for f in fits)} / 12.',
    f'Observed checkpoint/control screens: {len(screens)}; block trials: {len(blocks)}; final lanes: {len(final)}.','',
    'Screening uses two requests capped at 256 new tokens. It is selection evidence only.',
    'Final confirmation uses 16 reserved requests capped at 1,024 new tokens.',
    'Reference is FP32 greedy AR with TF32 disabled; drafter and source interface are BF16.','']
if comparison:
    lines+=['| Pair | Old decode TPS | Selected decode TPS | Old request TPS | Selected request TPS | Exact outputs |',
            '|---|---:|---:|---:|---:|---:|']
    for c in comparison:
        lines.append(f'| {c["pair"]} | {c["old_decode_tps"]:.2f} | {c["selected_decode_tps"]:.2f} | {c["old_request_tps"]:.2f} | {c["selected_request_tps"]:.2f} | 16/16 |')
else:
    lines.append('Final confirmation is not yet complete; no confirmed improvement is claimed.')
lines+=['','Training uses supplied Numina solution text, not generated rollouts. The new 4k control, 8k and 16k fits share the sampling recipe and validation split. Historical 4k training used a different recipe.','',
         'All candidates and fitting curves are retained in results.json. Exact output agreement is not an answer-quality metric.']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'fits':len(fits),'screens':len(screens),'blocks':len(blocks),'final':len(final),'comparison':comparison},indent=2))
