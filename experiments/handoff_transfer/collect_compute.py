"""Collect continuous-fit checkpoints without treating them as independent fits."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
from experiments.auf_vllm.compare_outputs import compare


def collect(root, family):
    signatures, cells = set(), []
    for kind, mode in (('fusion_r56', 'handoff-r56'), ('five_maps', 'handoff-five')):
        for steps in (500, 1000, 1500, 2000):
            module = root / 'modules' / f'full-{kind}' / (kind if steps == 2000 else f'seen_{steps * 8}')
            training = json.loads((module / 'summary.json').read_text())
            verified = json.loads((module / 'verification.json').read_text())
            assert training['optimizer_steps'] == verified['steps'] == steps
            assert training['processed_examples'] == steps * 8
            assert training['total_schedule_steps'] == 2000 and training['seed'] == 42
            assert verified['status'] == 'passed' and verified['frozen_non_fc_exact']
            assert verified['folded_relative_mse'] < 1e-6
            measured_root = root if steps == 2000 else root / 'compute' / f'updates-{steps}'
            repeats, hashes = [], {}
            for repeat in range(3):
                candidate = measured_root / f'measurements/{mode}-r{repeat}-w0.jsonl'
                reference = root / f'measurements/ar-r{repeat}-w0.jsonl'
                for path, expected_mode in ((candidate, mode), (reference, 'ar')):
                    summary = path.with_suffix('.summary.json')
                    contract = json.loads(summary.read_text())['contract']
                    assert contract['count'] == 128 and contract['cap'] == 2048
                    assert contract['family'] == family and contract['repeat'] == repeat
                    assert contract['mode'] == expected_mode
                    runtime = dict(contract['runtime_config']); runtime.pop('speculative_config', None)
                    signatures.add((contract['manifest_sha256'], json.dumps(runtime, sort_keys=True)))
                    for source in (path, summary):
                        hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
                result = compare([reference], [candidate])
                assert result['count'] == 128
                repeats.append(result)
            cells.append({'kind': kind, 'updates': steps, 'record_presentations': steps * 8,
                          'training': training, 'verification': verified,
                          'mean_tps': statistics.mean(r['method_tps'] for r in repeats),
                          'timing_stdev_tps': statistics.stdev(r['method_tps'] for r in repeats),
                          'mean_speedup_over_ar': statistics.mean(r['throughput_ratio'] for r in repeats),
                          'per_repeat': repeats, 'source_sha256': hashes})
    assert len(signatures) == 1, 'Runtime or evaluation manifest differs'
    return {'status': 'complete', 'family': family, 'cells': cells,
            'all_exact': all(r['exact_matches'] == r['finish_matches'] == 128 for c in cells for r in c['per_repeat']),
            'scope': '4096 distinct records, seed42, checkpoints from one 2000-update fit per architecture; three timing repetitions.',
            'cost_note': 'Cumulative AUF fitting time; add shared three-epoch ZIP initialization separately. These are not independently optimized training budgets.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--family', choices=('q8', 'q14'), required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); result = collect(args.root, args.family)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    assert result['all_exact'], 'Full token/finish mismatch; see report'
