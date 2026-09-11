"""Pair final CE/AUF results without confusing loss and initialization effects."""
import argparse
import json
import statistics
from pathlib import Path


def paired(results):
    groups={}
    for result in results:
        assert result['status']=='complete_verified'
        cell=result['cell'];key=(cell['family'],cell['kind'])
        assert cell['objective'] in ('ce','auf')
        group=groups.setdefault(key,{})
        assert cell['objective'] not in group, 'Duplicate loss result'
        group[cell['objective']]=result
    if not groups:raise ValueError('No final loss comparisons')
    rows=[]
    for (family,kind),group in sorted(groups.items()):
        assert set(group)=={'ce','auf'}, 'Both final losses required'
        ce,auf=group['ce'],group['auf']
        assert ce['gpu_hardware']==auf['gpu_hardware']
        assert ce['control_export_sha256']==auf['control_export_sha256']
        # Common hashed AR rows and summaries establish the same final requests,
        # runtime, timing repetitions and token reference via the strict collector.
        def ar_sources(result):
            return {p:h for p,h in result['sources'].items() if Path(p).name.startswith('ar-r')}
        assert len(ar_sources(ce))==6 and ar_sources(ce)==ar_sources(auf), 'Different AR reference evidence'
        for result in (ce,auf):
            fit=result['fit']
            assert fit['optimizer_steps']==2000 and fit['processed_examples']==16000
            assert fit['anchors_per_example']==512
            assert len(result['repeats'])==3
            assert all(r['count']==r['exact_matches']==r['finish_matches']==128 for r in result['repeats'])
        ratios=[]
        for c,a in zip(ce['repeats'],auf['repeats']):
            assert c['method_output_tokens']==a['method_output_tokens']
            assert c['method_tps']>0 and a['method_tps']>0
            ratios.append(a['method_tps']/c['method_tps'])
        rows.append(dict(family=family,interface=kind,ce_tps=ce['mean_tps'],auf_tps=auf['mean_tps'],
            auf_over_ce_by_repeat=ratios,mean_auf_over_ce=statistics.mean(ratios),
            ce_lr=ce['cell']['lr'],auf_lr=auf['cell']['lr'],
            interpretation='Separately LR-tuned loss recipes with the same interface/initializer and compute budget; not fixed-LR causal isolation',
            uncertainty_scope='Three timing repetitions and one fitting seed; no fitting-seed confidence interval'))
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    rows=paired([json.loads(p.read_text()) for p in args.results])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(rows,indent=2)+'\n')
