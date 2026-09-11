"""Report standalone Transformers replication against its own AR baseline."""
import argparse
import json
from pathlib import Path
from experiments.auf_vllm.compare_outputs import compare
from experiments.handoff_transfer.freeze_confirmation import digest


def collect(root, primary, family):
    methods = ('ar', 'normal', 'zip', 'handoff-r56', 'handoff-five')
    signatures, hashes, phases = set(), {}, {}
    for phase, count, cap in (('gate', 4, 128), ('full', 128, 2048)):
        results = {}
        for mode in methods:
            path = root / phase / f'{mode}-r0-w0.jsonl'
            summary_path = path.with_suffix('.summary.json')
            summary = json.loads(summary_path.read_text()); contract = summary['contract']
            assert summary['timing_valid'] and contract['runtime_config']['backend'] == 'transformers'
            assert contract['family'] == family and contract['mode'] == mode and contract['repeat'] == 0
            assert contract['count'] == count and contract['cap'] == cap
            signatures.add((contract['manifest_sha256'], json.dumps(contract['runtime_config'], sort_keys=True)))
            source = primary / f'{mode}-r0-w0.summary.json'
            expected = json.loads(source.read_text())['contract']
            assert expected['family'] == family and expected['mode'] == mode
            assert contract['export_sha256'] == expected['export_sha256'], 'Backend replication used another checkpoint'
            assert contract['manifest_sha256'] == expected['manifest_sha256']
            results[mode] = compare([root / phase / 'ar-r0-w0.jsonl'], [path])
            assert results[mode]['count'] == count
            for item in (path, summary_path, source):
                hashes[str(item)] = digest(item)
        phases[phase] = results
    assert len(signatures) == 1, 'Transformers runtime or manifest mismatch'
    for mode, result in phases['full'].items():
        result['ratio_to_normal'] = result['method_tps'] / phases['full']['normal']['method_tps']
        result['ratio_to_zip'] = result['method_tps'] / phases['full']['zip']['method_tps']
    return {'status': 'complete', 'family': family, 'backend': 'transformers',
            'all_exact': all(r['count'] == r['exact_matches'] == r['finish_matches'] for p in phases.values() for r in p.values()),
            'phases': phases, 'source_sha256': hashes,
            'scope': 'Secondary backend replication: one timing repetition, seed42 checkpoints,128 development requests/cap2048. Speedups use Transformers AR only; no cross-backend TPS ratio.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'primary', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--family', choices=('q8', 'llama'), required=True)
    a = p.parse_args(); report = collect(a.root, a.primary, a.family)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    assert report['all_exact'], 'Transformers output mismatch; diagnostics retained'
