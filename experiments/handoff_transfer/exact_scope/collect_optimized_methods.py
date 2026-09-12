"""Independently audit the complete matched optimized-runtime comparison."""
import argparse
import json
from pathlib import Path

from experiments.auf_vllm.compare_outputs import compare
from .collect_single_request import read,rows,summarize_requests
from .single_request import LABELS,CAMPAIGN
from .optimized_methods import NEW_CAMPAIGN


def validate_runtime(summary,reference):
    c,a=summary['contract'],reference['contract']
    for value in (c,a):
        assert value['runtime_profile']=='optimized' and value['batch_invariant'] is False
        assert value['request_batch_size']==1
    def common(config):return {k:v for k,v in config.items() if k!='speculative_config'}
    assert common(c['runtime_config'])==common(a['runtime_config']),'Unmatched engine arguments'
    e=summary['effective_runtime']
    assert e==reference['effective_runtime'],'Unmatched effective runtime'
    assert e['compilation_mode']>0 and e['optimization_level']==3
    assert e['async_scheduling'] is True and e['batch_invariant']=='0'
    assert e['use_v2_model_runner'] is True and e['quantization'] is None
    assert e['dtype']=='torch.bfloat16' and e['tensor_parallel_size']==1
    assert 'NONE' not in e['cudagraph_mode']


def collect(root):
    results=[];issues=[];progress=[]
    for family in ('q8','cross'):
        cohort=read(root/'exact32e1b8-results'/family/'evaluation/eval.json')
        collected={}
        for label in LABELS:
            mode=label if label in ('ar','native') else 'matrix'
            paths=[];references=[];devices=[];effects=[]
            try:
                for worker in range(4):
                    folder=root/NEW_CAMPAIGN/family/f'worker-{worker}'
                    path=folder/label/f'{mode}-r0-w{worker}.jsonl'
                    ar=folder/'ar'/f'ar-r0-w{worker}.jsonl'
                    s,a=read(path.with_suffix('.summary.json')),read(ar.with_suffix('.summary.json'))
                    validate_runtime(s,a)
                    for k,v in dict(family=family,count=128,cap=2048,worker_index=worker,workers=4,repeat=0).items():
                        assert s['contract'][k]==a['contract'][k]==v
                    assert s['contract']['manifest_sha256']==a['contract']['manifest_sha256']
                    uuid=s['gpu_after'][0]['device_uuid']
                    assert all(x[k][0]['device_uuid']==uuid for x in (s,a) for k in ('gpu_before','gpu_after'))
                    assert all(x['timing_valid'] and all(v['passed'] for v in x['asset_checks']) for x in (s,a))
                    if label=='native':
                        prior=read(root/CAMPAIGN/family/f'worker-{worker}'/'native'/f'native-r0-w{worker}.summary.json')['contract']
                        for key in ('native_draft_sha256','native_config_sha256'):
                            assert s['contract'][key] and s['contract'][key]==prior[key]
                    elif label!='ar':
                        oldlabel='original' if label=='original' else label+'-lr0.0001'
                        expected=read(root/'exact32e1b8-results'/family/oldlabel/'matrix-r0-w0-b128.summary.json')['contract']['export_sha256']
                        assert s['contract']['export_sha256']==expected
                    if label!='ar':
                        spec=s['contract']['runtime_config']['speculative_config']
                        assert spec['method']=='dflash'
                        assert spec['num_speculative_tokens']==(9 if family=='cross' and label=='native' else 15)
                    actual=rows(path)
                    assert [(r['group_id'],r['prompt_ids']) for r in actual]==[(r['group_id'],r['prompt_token_ids']) for r in cohort[worker::4]]
                    assert compare([ar],[path])['count']==32
                    paths.append(path);references.append(ar);devices.append(uuid);effects.append(s['effective_runtime'])
                records=[r for p in paths for r in rows(p)]
                assert len(records)==len({r['group_id'] for r in records})==128
                assert len(set(devices))==4
                assert all(e==effects[0] for e in effects)
                comparison=compare(references,paths)
                stats=summarize_requests(records)
                if label!='ar':assert stats['draft_blocks']>0,'No measured speculative iterations'
                collected[label]=dict(family=family,label=label,physical_gpus=devices,effective_runtime=effects[0],
                    exact_ar=comparison['exact_matches'],finish_matches=comparison['finish_matches'],
                    observed_bitwise_equivalence=comparison['exact_matches']==128,
                    output_comparison=comparison,**stats)
            except (OSError,KeyError,AssertionError,ValueError) as error:
                issues.append(dict(family=family,label=label,error=str(error)))
            progress.append(dict(family=family,label=label,verified_shards=len(paths),required_shards=4))
        for label,r in collected.items():
            for baseline in ('ar','native','original'):
                if baseline in collected:
                    b=collected[baseline]
                    r['tps_ratio_'+baseline]=r['output_tps']/b['output_tps']
                    r['request_time_ratio_'+baseline]=b['total_request_seconds']/r['total_request_seconds']
            results.append(r)
    return dict(status='complete' if len(results)==14 and not issues else 'incomplete',
        verified_cells=len(results),expected_cells=14,rows=results,issues=issues,progress=progress,
        scope='Fresh AR and all six speculative methods, same O3/async/non-invariant runtime, BF16, batch1, 128 identical prompts per target, cap2048, same physical GPUs per shard. Native Llama block10, other speculative arms block16. One timing pass.',
        interpretation='TPS ratios compare pooled output tokens/summed request wall time. If sequences differ, request-time ratios include output-length effects and are not identical-output latency speedups. Exact agreement is measured, not assumed. Engine stages are not isolated GPU kernel durations.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    report=collect(a.root);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
    lines=[report['scope'],'',report['interpretation'],'',
           '| Target | Method | TPS | Mean request s | TPS / AR | Time ratio / AR | TPS / native | Accepted proposals/block | Exact AR |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in report['rows']:
        def val(k):return '-' if r.get(k) is None else f'{r[k]:.3f}'
        lines.append(f"| {r['family']} | {r['label']} | {r['output_tps']:.2f} | {val('mean_request_seconds')} | {val('tps_ratio_ar')} | {val('request_time_ratio_ar')} | {val('tps_ratio_native')} | {val('accepted_proposals_per_block')} | {r['exact_ar']}/128 |")
    a.out.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'],verified_cells=report['verified_cells'],issues=report['issues'])))
    if a.require_complete and report['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
