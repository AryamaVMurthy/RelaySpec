"""Summarize completed AUF timing repetitions against normal, without AR claims."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
from experiments.auf_vllm.compare_outputs import compare


def collect(normal, candidate, repetitions):
    assert 1 <= repetitions <= 3
    results, signatures, hashes = {}, set(), {}
    for mode in ('handoff-r56', 'handoff-five'):
        rows = []
        for repeat in range(repetitions):
            base = normal / f'normal-r{repeat}-w0.jsonl'
            method = candidate / f'{mode}-r{repeat}-w0.jsonl'
            for path, expected_mode in ((base, 'normal'), (method, mode)):
                summary = path.with_suffix('.summary.json')
                contract = json.loads(summary.read_text())['contract']
                assert contract['mode'] == expected_mode and contract['repeat'] == repeat
                assert contract['family'] == 'q8' and contract['count'] == 128 and contract['cap'] == 2048
                config = dict(contract['runtime_config']); config.pop('speculative_config', None)
                signatures.add((contract['manifest_sha256'], json.dumps(config, sort_keys=True)))
                for source in (path, summary):
                    hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
            row = compare([base], [method])
            assert row['count'] == row['exact_matches'] == row['finish_matches'] == 128
            row['normal_tps'] = row.pop('ar_tps')
            row['normal_output_tokens'] = row.pop('ar_output_tokens')
            row['repeat'] = repeat
            rows.append(row)
        results[mode] = {'per_repeat': rows,
            'mean_tps': statistics.mean(r['method_tps'] for r in rows),
            'timing_stdev_tps': statistics.stdev(r['method_tps'] for r in rows) if repetitions > 1 else None,
            'mean_ratio_to_normal': statistics.mean(r['throughput_ratio'] for r in rows)}
    assert len(signatures) == 1, 'Runtime or evaluation manifest differs'
    return {'status': 'partial_primary_comparison', 'timing_repetitions_complete': repetitions,
            'results': results, 'source_sha256': hashes,
            'scope': 'Qwen8, one fitting seed, 128 development requests, cap2048, natural EOS. '
                     'Agreement is with normal RelaySpec, not yet AR. ZIP/native/CE comparisons pending. '
                     'Timing repetition spread does not measure fitting-seed uncertainty.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--normal', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--repetitions', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); result = collect(a.normal, a.candidate, a.repetitions)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['results'], indent=2))
