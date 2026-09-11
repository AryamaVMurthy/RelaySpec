"""Aggregate independent fits after averaging their timing repetitions."""
import argparse
import json
import statistics
from pathlib import Path
from experiments.handoff_transfer.collect_transfer import main as collect_one

METHODS = ('normal', 'zip', 'handoff-r56', 'handoff-five')


def summarize(reports):
    assert set(reports) == {42, 43, 44}
    result = {}
    for mode in METHODS:
        result[mode] = {}
        for metric in ('mean_tps', 'mean_speedup_over_ar', 'mean_tps_ratio_to_normal', 'mean_tps_ratio_to_zip'):
            values = [reports[s]['rows'][mode][metric] for s in (42, 43, 44)]
            result[mode][metric] = {'seed_values': values, 'mean': statistics.mean(values),
                'fitting_seed_stdev': statistics.stdev(values), 'min': min(values), 'max': max(values)}
    return result


def main(a):
    reports, provenance = {}, {}
    baseline = a.root / 'q8'
    for seed in (42, 43, 44):
        run = baseline if seed == 42 else a.root / f'seeds/q8-s{seed}'
        transfer = json.loads((run / 'transfer.json').read_text())
        initializer = Path(transfer['base_export']).parents[1]
        zip_fit = json.loads((initializer / 'summary.json').read_text())
        assert zip_fit['config']['seed'] == seed and zip_fit['config']['records'] == 4096
        assert zip_fit['config']['epochs'] == 3
        normal = json.loads((run / 'normal/summary.json').read_text())
        assert normal['status'] == 'complete' and normal['contract']['seed'] == seed
        assert normal['contract']['records'] == 4096 and normal['contract']['epochs'] == 3
        fits = {}
        for kind in ('fusion_r56', 'five_maps'):
            path = run / f'modules/full-{kind}/{kind}'
            training = json.loads((path / 'summary.json').read_text())
            verified = json.loads((path / 'verification.json').read_text())
            assert training['seed'] == seed and training['optimizer_steps'] == verified['steps'] == 2000
            assert training['processed_examples'] == 16000 and verified['status'] == 'passed'
            assert verified['frozen_non_fc_exact']
            fits[kind] = {'training': training, 'verification': verified}
        expected_exports = {'zip': transfer['base_sha256'], 'normal': normal['export_sha256'],
            'handoff-r56': fits['fusion_r56']['verification']['export_sha256'],
            'handoff-five': fits['five_maps']['verification']['export_sha256']}
        for mode, digest in expected_exports.items():
            for repeat in range(3):
                measured = json.loads((run / f'measurements/{mode}-r{repeat}-w0.summary.json').read_text())
                assert measured['contract']['export_sha256'] == digest, 'Evaluation used a different checkpoint'
        out = a.out.parent / f'q8-seed-{seed}-comparison.json'
        collect_one(argparse.Namespace(root=run, family='q8', out=out, training_seed=seed,
            methods=['ar', *METHODS], method_roots={'ar': baseline / 'measurements'}))
        reports[seed] = json.loads(out.read_text())
        provenance[seed] = {'zip': zip_fit, 'normal': normal, 'auf': fits,
                            'comparison_path': str(out)}
    result = {'status': 'complete', 'family': 'q8', 'training_seeds': [42, 43, 44],
        'timing_repetitions_per_seed': 3, 'results': summarize(reports), 'per_seed': reports,
        'training_provenance': provenance,
        'scope': 'Average three timing repetitions within each fit before calculating variation across three fitting seeds. Fixed records and frozen rollout cache; not independent data draws or untouched confirmation. AR measurements are shared references, not nine independent AR measurements.'}
    a.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
