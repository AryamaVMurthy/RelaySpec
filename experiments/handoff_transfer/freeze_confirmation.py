"""Pin primary checkpoints only after complete development and seed evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def freeze(root, reports, workload_reports, out, selected_families=('q8', 'q14', 'llama')):
    assert not out.exists(), 'Never overwrite a frozen confirmation protocol'
    assert selected_families and set(selected_families) <= {'q8', 'q14', 'llama'}
    evidence, families = {}, {}
    for family in selected_families:
        required = [reports / f'{family}-comparison.json']
        required += [workload_reports / f'{family}-{w}.json' for w in ('math', 'gsm', 'code', 'chat')]
        if family in ('q8', 'llama'):
            required += [reports / f'{family}-seeds-comparison.json']
        for path in required:
            report = json.loads(path.read_text())
            assert report['status'] == 'complete' and report['family'] == family
            if 'seeds-comparison' in path.name:
                assert report['training_seeds'] == [42, 43, 44]
                assert set(report['per_seed']) == {'42', '43', '44'}
                assert report['timing_repetitions_per_seed'] == 3
                assert all(r['all_exact'] for r in report['per_seed'].values())
            else:
                assert report['all_exact'] and report['timing_repetitions'] == 3
            evidence[str(path)] = digest(path)
        methods = ('ar', 'normal', 'zip', 'handoff-r56', 'handoff-five')
        if family == 'q8':
            methods += ('native',)
        contracts = {}
        for mode in methods:
            source = root / family / f'measurements/{mode}-r0-w0.summary.json'
            summary = json.loads(source.read_text())
            contract = summary['contract']
            assert summary['timing_valid'] and contract['family'] == family and contract['mode'] == mode
            assert contract['count'] == 128 and contract['cap'] == 2048
            checkpoint = contract['runtime_config'].get('speculative_config', {}).get('model')
            if checkpoint:
                checkpoint = Path(checkpoint)
                measured_hash = contract['export_sha256']
                current_hash = digest(checkpoint / 'model.safetensors')
                if mode != 'native':
                    assert measured_hash == current_hash, 'Checkpoint changed since development evaluation'
                contracts[mode] = {'checkpoint': str(checkpoint), 'weights_sha256': current_hash,
                    'config_sha256': digest(checkpoint / 'config.json'), 'runtime_config': contract['runtime_config']}
            else:
                assert mode == 'ar'
                contracts[mode] = {'runtime_config': contract['runtime_config']}
            evidence[str(source)] = digest(source)
        families[family] = contracts
    result = {'status': 'frozen', 'families': families, 'development_evidence_sha256': evidence,
        'evaluation': {'workloads': ['math', 'gsm', 'code', 'chat'], 'requests_per_workload': 128,
                       'max_output_tokens': 2048, 'timing_repetitions': 3, 'natural_eos': True},
        'selection': 'All prespecified primary seed42 arms retained; no winner selected using confirmation outputs.',
        'scope': 'Confirmation of the primary 4096-record endpoints. Exploratory data/compute/rank points are not independently confirmed by this protocol.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'reports', 'workload-reports', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--families', nargs='+', choices=('q8', 'q14', 'llama'), default=['q8', 'q14', 'llama'])
    a = p.parse_args(); freeze(a.root, a.reports, a.workload_reports, a.out, a.families)
