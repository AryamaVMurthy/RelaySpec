"""Collect complete matched-data scaling repetitions against shared AR controls."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from experiments.auf_vllm.compare_outputs import compare


def collect(root, baseline):
    transfer = json.loads((root / 'transfer.json').read_text())
    results, hashes, signatures = {}, {}, set()
    for mode in ('zip', 'handoff-r56', 'handoff-five'):
        repeats = []
        for repeat in range(3):
            ar = baseline / f'ar-r{repeat}-w0.jsonl'
            method = root / f'measurements/{mode}-r{repeat}-w0.jsonl'
            for path in (ar, method):
                summary_path = path.with_suffix('.summary.json')
                summary = json.loads(summary_path.read_text())
                contract = summary['contract']
                assert contract['count'] == 128 and contract['cap'] == 2048
                assert contract['repeat'] == repeat and contract['family'] == transfer['family']
                config = dict(contract['runtime_config'])
                config.pop('speculative_config', None)
                signatures.add((contract['manifest_sha256'], json.dumps(config, sort_keys=True)))
                for source in (path, summary_path):
                    hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
            row = compare([ar], [method])
            assert row['count'] == 128
            repeats.append(row)
        results[mode] = repeats
    assert len(signatures) == 1, 'Runtime or evaluation manifest mismatch'
    rows = {}
    for mode, repeats in results.items():
        rows[mode] = {
            'mean_tps': statistics.mean(r['method_tps'] for r in repeats),
            'timing_stdev_tps': statistics.stdev(r['method_tps'] for r in repeats),
            'mean_speedup_over_ar': statistics.mean(r['throughput_ratio'] for r in repeats),
            'mean_ratio_to_matched_zip': statistics.mean(r['method_tps'] / results['zip'][i]['method_tps'] for i, r in enumerate(repeats)),
            'token_matches': [r['exact_matches'] for r in repeats],
            'finish_matches': [r['finish_matches'] for r in repeats],
        }
    exact = all(r['exact_matches'] == r['finish_matches'] == 128 for rs in results.values() for r in rs)
    return {'status': 'complete', 'all_exact': exact, 'records': transfer['records'],
            'initialization_training_seconds': transfer['initialization_training_seconds'],
            'rows': rows, 'per_repeat': results, 'source_sha256': hashes,
            'scope': 'One fitting seed; three timing repetitions; development evaluation, not untouched confirmation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'baseline', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.root, args.baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['rows']), flush=True)
    assert result['all_exact'], 'Full token/finish agreement failed; see report'
