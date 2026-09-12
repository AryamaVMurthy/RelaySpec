"""Recompute old-method TPS ratios against optimized AR; audit output drift."""
import argparse
import json
from pathlib import Path

from experiments.auf_vllm.compare_outputs import compare
from .collect_single_request import rows, read, summarize_requests
from .single_request import CAMPAIGN, LABELS
from .optimized_ar import NEW_CAMPAIGN


def collect(root):
    output = []
    issues = []
    for family in ('q8', 'cross'):
        try:
            baseline_paths = []
            baseline_records = []
            for worker in range(4):
                path = root / NEW_CAMPAIGN / family / f'worker-{worker}' / f'ar-r0-w{worker}.jsonl'
                s = read(path.with_suffix('.summary.json'))
                c = s['contract']
                assert c['runtime_profile'] == 'optimized-ar' and c['batch_invariant'] is False
                assert c['request_batch_size'] == 1 and c['count'] == 128 and c['cap'] == 2048
                assert c['workers'] == 4 and c['worker_index'] == worker
                assert c['runtime_config']['optimization_level'] == 3
                assert c['runtime_config']['async_scheduling'] is True
                assert 'compilation_config' not in c['runtime_config']
                assert all(x['passed'] for x in s['asset_checks']) and s['timing_valid']
                assert s['gpu_before'][0]['device_uuid'] == s['gpu_after'][0]['device_uuid']
                actual = rows(path)
                cohort = read(root / 'exact32e1b8-results' / family / 'evaluation/eval.json')[worker::4]
                assert [(r['group_id'], r['prompt_ids']) for r in actual] == [(r['group_id'],r['prompt_token_ids']) for r in cohort]
                assert len(actual) == 32 and all(r['timing_valid'] for r in actual)
                baseline_paths.append(path)
                baseline_records.extend(actual)
            assert len({r['group_id'] for r in baseline_records}) == 128
            baseline = summarize_requests(baseline_records)
            output.append(dict(family=family, label='optimized_ar', **baseline,
                               speedup_vs_optimized_ar=1.0, request_time_ratio_vs_optimized_ar=1.0,
                               exact_optimized_ar=128))
            for label in LABELS:
                method_paths = []
                records = []
                for worker, base in enumerate(baseline_paths):
                    mode = label if label in ('ar','native') else 'matrix'
                    path = root / CAMPAIGN / family / f'worker-{worker}' / label / f'{mode}-r0-w{worker}.jsonl'
                    a, b = read(base.with_suffix('.summary.json')), read(path.with_suffix('.summary.json'))
                    assert a['gpu_after'][0]['device_uuid'] == b['gpu_after'][0]['device_uuid']
                    for key in ('manifest_sha256','count','cap','worker_index','workers'):
                        assert a['contract'][key] == b['contract'][key]
                    assert a['contract']['runtime_config']['model'] == b['contract']['runtime_config']['model']
                    assert all(x['passed'] for x in b['asset_checks']) and b['timing_valid']
                    rs = rows(path)
                    assert [(r['group_id'],r['prompt_ids']) for r in rs] == [(r['group_id'],r['prompt_ids']) for r in rows(base)]
                    method_paths.append(path)
                    records.extend(rs)
                comparison = compare(baseline_paths, method_paths)
                assert comparison['count'] == 128
                stats = summarize_requests(records)
                output.append(dict(family=family,label='previous_ar' if label=='ar' else label,**stats,
                    speedup_vs_optimized_ar=stats['output_tps']/baseline['output_tps'],
                    request_time_ratio_vs_optimized_ar=baseline['total_request_seconds']/stats['total_request_seconds'],
                    exact_optimized_ar=comparison['exact_matches'], finish_matches=comparison['finish_matches'],
                    output_comparison=comparison))
        except (OSError, KeyError, AssertionError, ValueError) as error:
            issues.append(dict(family=family,error=str(error)))
    return dict(status='complete' if len(output)==16 and not issues else 'incomplete', rows=output,issues=issues,
        scope='Existing invariant speculative measurements divided by new O3, async, non-invariant AR TPS. Same128 prompts and physical GPUs, batch1 cap2048 BF16. Different runtime configurations; no speculative remeasurement.',
        caveat='If outputs differ, TPS ratios are per-token throughput comparisons, not identical-output latency speedups. Request-time ratios may additionally reflect changed output lengths. O3 is the highest provided optimization level, not a guarantee of globally optimal tuning.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--require-complete',action='store_true');args=p.parse_args()
    report=collect(args.root);args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    lines=[report['scope'],'',report['caveat'],'',
           '| Target | Method | TPS | / optimized AR TPS | Request time ratio | Exact optimized AR |',
           '|---|---|---:|---:|---:|---:|']
    for r in report['rows']:
        lines.append(f"| {r['family']} | {r['label']} | {r['output_tps']:.2f} | {r['speedup_vs_optimized_ar']:.3f} | {r['request_time_ratio_vs_optimized_ar']:.3f} | {r['exact_optimized_ar']}/128 |")
    args.out.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'],rows=len(report['rows']),issues=report['issues'])))
    if args.require_complete and report['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
