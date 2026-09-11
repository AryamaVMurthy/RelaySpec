"""Only complete, matched 128/cap2048 matrix measurements enter final results."""
import argparse,json,statistics
from pathlib import Path
from experiments.auf_vllm.compare_outputs import compare
from experiments.handoff_transfer.matrix.prepare import digest

def collect(run,baseline):
    cell=json.loads((run/'cell.json').read_text())
    kind=cell['kind'];tag='steps-2000'
    fit=json.loads((run/'modules'/tag/kind/'summary.json').read_text())
    verification=json.loads((run/'modules'/tag/kind/'verification.json').read_text())
    export=run/'exports'/tag/kind/'model.safetensors'
    assert fit['optimizer_steps']==2000 and fit['processed_examples']==16000
    assert fit['anchors_per_example']==512 and fit['objective']==cell['objective']
    assert verification['status']=='passed' and verification['export_sha256']==digest(export)
    results=[];sources={};controls={'normal':[],'zip':[]}
    for repeat in range(3):
        method=run/'measurements'/f'matrix-r{repeat}-w0.jsonl'
        ar=baseline/'measurements'/f'ar-r{repeat}-w0.jsonl'
        m=json.loads(method.with_suffix('.summary.json').read_text())['contract']
        a=json.loads(ar.with_suffix('.summary.json').read_text())['contract']
        assert m['count']==a['count']==128 and m['cap']==a['cap']==2048
        assert m['repeat']==a['repeat']==repeat and m['family']==a['family']==cell['family']
        assert m['manifest_sha256']==a['manifest_sha256']
        assert m['export_sha256']==verification['export_sha256']
        mr=dict(m['runtime_config']);arconfig=dict(a['runtime_config'])
        mr.pop('speculative_config',None);arconfig.pop('speculative_config',None)
        assert mr==arconfig,'Unmatched runtime contract'
        result=compare([ar],[method])
        assert result['count']==result['exact_matches']==result['finish_matches']==128
        results.append(result)
        for control in controls:
            path=baseline/'measurements'/f'{control}-r{repeat}-w0.jsonl'
            c=json.loads(path.with_suffix('.summary.json').read_text())['contract']
            assert c['count']==128 and c['cap']==2048 and c['repeat']==repeat
            assert c['manifest_sha256']==m['manifest_sha256'] and c['family']==cell['family']
            runtime=dict(c['runtime_config']);runtime.pop('speculative_config',None)
            assert runtime==mr,'Unmatched control runtime'
            comparison=compare([path],[method])
            assert comparison['exact_matches']==comparison['finish_matches']==128
            controls[control].append(comparison['throughput_ratio'])
            sources[str(path)]=digest(path);sources[str(path.with_suffix('.summary.json'))]=digest(path.with_suffix('.summary.json'))
        for p in [ar,method,ar.with_suffix('.summary.json'),method.with_suffix('.summary.json')]:sources[str(p)]=digest(p)
    tps=[r['method_tps'] for r in results]
    return {'status':'complete_verified','cell':cell,'fit':fit,'verification':verification,
        'ratio_to_controls':{k:statistics.mean(v) for k,v in controls.items()},'control_ratios_by_repeat':controls,
        'repeats':results,'mean_tps':statistics.mean(tps),'timing_std_tps':statistics.stdev(tps),
        'mean_paired_ar_speedup':statistics.mean(r['throughput_ratio'] for r in results),
        'uncertainty_scope':'three timing repetitions, one fitting seed; not fitting-seed uncertainty',
        'sources':sources}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();result=collect(a.run,a.baseline)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
