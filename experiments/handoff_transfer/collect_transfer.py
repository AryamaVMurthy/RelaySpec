"""Aggregate completed matched runs; never promote partial logs to evidence."""
import argparse,json,hashlib,statistics
from pathlib import Path
from experiments.auf_vllm.compare_outputs import compare

def main(a):
    root=a.root/'measurements'
    methods=['ar','zip','handoff-r56','handoff-five']
    files={m:[root/f'{m}-r{r}-w0.jsonl' for r in range(3)] for m in methods}
    missing=[str(p) for paths in files.values() for p in paths if not p.exists() or not p.with_suffix('.summary.json').exists()]
    if missing:raise RuntimeError(f'Incomplete evaluation, missing {missing}')
    reports={}
    signatures=[]
    hashes={}
    for m,paths in files.items():
        reports[m]=[]
        for r,p in enumerate(paths):
            summary=json.loads(p.with_suffix('.summary.json').read_text())
            config=dict(summary['contract']['runtime_config']);config.pop('speculative_config',None)
            signatures.append((summary['contract']['manifest_sha256'],json.dumps(config,sort_keys=True)))
            assert summary['contract']['cap']==2048 and summary['contract']['count']==128
            result=compare([files['ar'][r]],[p])
            assert result['count']==128
            result['repeat']=r
            reports[m].append(result)
            for source in [p,p.with_suffix('.summary.json')]:hashes[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(set(signatures))==1,'Runtime or request manifest mismatch'
    rows={}
    for m,results in reports.items():
        tps=[v['method_tps'] for v in results]
        rows[m]={'mean_tps':statistics.mean(tps),'timing_repetition_stdev_tps':statistics.stdev(tps),
                 'mean_speedup_over_ar':statistics.mean(v['throughput_ratio'] for v in results),
                 'mean_tps_ratio_to_zip':statistics.mean(v['method_tps']/reports['zip'][r]['method_tps'] for r,v in enumerate(results)),
                 'exact_token_matches':[v['exact_matches'] for v in results],
                 'exact_finish_matches':[v['finish_matches'] for v in results]}
    result={'family':a.root.name,'status':'complete','evaluation':'128 Numina development requests, max2048, natural EOS, greedy',
            'training_seeds':[42],'timing_repetitions':3,'rows':rows,'per_repeat':reports,'source_sha256':hashes,
            'normal_relayspec_control':'not yet present; ZIP is a separate baseline',
            'uncertainty':'Timing repetition variation only; not fitting-seed uncertainty; not untouched confirmation.'}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
