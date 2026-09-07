"""Audit paired transfer benchmarks and report request-bootstrap uncertainty."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--measurements', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--count', type=int, default=128)
    ap.add_argument('--cap', type=int, default=2048)
    ap.add_argument('--repeat', type=int, default=0, help='Speculative modes repeat index; AR baseline remains repeat0')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    modes = ['ar8', 'native8', 'mapped']
    groups, sources = {}, {}
    for mode in modes:
        repeat = 0 if mode == 'ar8' else args.repeat
        files = sorted(args.measurements.glob(f'worker_*/{mode}-r{repeat}.jsonl'))
        rr = [json.loads(line) for path in files for line in path.read_text().splitlines()]
        assert len(rr) == args.count, (mode, len(rr), args.count)
        groups[mode] = {r['group_id']: r for r in rr}
        assert len(groups[mode]) == args.count
        sources[mode] = []
        for path in files:
            meta = json.loads(path.with_suffix('.summary.json').read_text())
            assert meta['timing_valid'] and meta['args']['cap'] == args.cap
            assert meta['config']['enable_prefix_caching'] is False
            sources[mode].append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                  'job': meta['slurm_job_id'], 'gpu': meta['cuda_visible_devices']})
        for r in rr:
            assert r['timing_valid'] and r['wall_seconds'] > 0
            assert r['output_tokens'] == len(r['output_ids']) <= args.cap
    keys = sorted(groups['ar8'])
    assert all(sorted(g) == keys for g in groups.values())
    refs = {r['group_id']: r for r in map(json.loads, args.reference.read_text().splitlines())}
    mismatches, historical = [], []
    for k in keys:
        a, b, c = [groups[m][k] for m in modes]
        assert a['prompt_ids'] == b['prompt_ids'] == c['prompt_ids'] == refs[k]['prompt_ids']
        if not (a['output_ids'] == b['output_ids'] == c['output_ids'] and
                a['finish_reason'] == b['finish_reason'] == c['finish_reason']):
            mismatches.append(k)
        if any(r['output_ids'] != refs[k]['output_ids'][:args.cap] for r in (a, b, c)):
            historical.append(k)
    times = {m: np.array([groups[m][k]['wall_seconds'] for k in keys]) for m in modes}
    tokens = {m: np.array([groups[m][k]['output_tokens'] for k in keys]) for m in modes}
    stats = {m: {'tps': float(tokens[m].sum() / times[m].sum()), 'tokens': int(tokens[m].sum()),
                 'seconds': float(times[m].sum()),
                 'length_stops': sum(groups[m][k]['finish_reason'] == 'length' for k in keys)} for m in modes}
    # Same request resamples across modes. This interval does not estimate repeated-run variability.
    ix = np.random.default_rng(42).integers(0, len(keys), size=(10000, len(keys)))
    ratios = {}
    for numerator, denominator in [('mapped', 'ar8'), ('native8', 'ar8'), ('mapped', 'native8')]:
        sampled = (tokens[numerator][ix].sum(1) / times[numerator][ix].sum(1)) / (tokens[denominator][ix].sum(1) / times[denominator][ix].sum(1))
        ratios[f'{numerator}_over_{denominator}'] = {
            'ratio': stats[numerator]['tps'] / stats[denominator]['tps'],
            'paired_request_95ci': np.quantile(sampled, [.025, .975]).tolist()}
    result = {'count': args.count, 'cap': args.cap, 'speculative_repeat': args.repeat, 'all_tokens_and_stops_equal': not mismatches,
              'different_groups': mismatches, 'different_from_historical_output': historical,
              'stats': stats, 'ratios': ratios, 'sources': sources,
              'uncertainty_scope': '10000 paired request bootstrap samples, seed42; excludes fitting seeds, repeated timing runs and hardware variation.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    assert not mismatches, 'Token mismatch: investigate before claiming equivalent-work speedups'


if __name__ == '__main__':
    main()
