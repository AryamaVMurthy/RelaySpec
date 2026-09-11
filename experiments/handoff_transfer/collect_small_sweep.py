"""Collect the entire fixed-update small-data sweep without selecting winners."""
import argparse
import json
from pathlib import Path
from experiments.handoff_transfer.collect_scaling import collect


def main(args):
    jobs = json.loads(args.jobs.read_text())
    fits = [{'records': 16, 'job_id': '31608'}, *jobs['fits']]
    assert [x['records'] for x in fits] == [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    cells = []
    for fit in fits:
        root = args.root / f"scaling/q8-n{fit['records']}-u1024-{fit['job_id']}"
        result = collect(root, args.root / 'q8/measurements')
        assert result['records'] == fit['records'] and result['all_exact']
        training = {}
        for kind in ('fusion_r56', 'five_maps'):
            summary = json.loads((root / f'modules/full-{kind}/{kind}/summary.json').read_text())
            verification = json.loads((root / f'modules/full-{kind}/{kind}/verification.json').read_text())
            assert summary['optimizer_steps'] == verification['steps'] == 1024
            assert summary['processed_examples'] == 8192 and verification['status'] == 'passed'
            training[kind] = summary
        result['training'] = training
        cells.append(result)
    report = {'status': 'complete', 'all_exact': True, 'auf_updates': 1024,
              'auf_record_presentations': 8192, 'cells': cells,
              'budget_note': 'AUF work fixed. ZIP initialization uses three epochs on each data size; '
                             'its compute varies with N and must be charged separately.',
              'scope': 'One training seed; development requests; timing variation only.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print('VERIFIED_SMALL_DATA_SWEEP', len(cells), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('jobs', 'root', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    main(p.parse_args())
