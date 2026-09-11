"""Queue rank ablations only after the rank8 end-to-end gate succeeds."""
import json
import subprocess
from pathlib import Path


def main():
    control = Path('/home/aryama.murthy/relayspec-auf-20260911')
    path = control / 'outputs/handoff-transfer/capacity-jobs.json'
    assert not path.exists(), 'Inspect existing submission record; do not duplicate jobs'
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {'rank56': 'Reuse main Q8 handoff-r56 fit and measurements', 'jobs': [], 'status': 'submitting'}
    path.write_text(json.dumps(report, indent=2) + '\n')
    previous = '31635'
    for rank in (16, 32, 128, 256):
        for phase in ('fit', 'eval'):
            job = subprocess.check_output(['sbatch', '--parsable', f'--dependency=afterok:{previous}',
                f'--export=ALL,AUF_CAPACITY_RANK={rank}',
                str(control / f'experiments/handoff_transfer/capacity_{phase}.sbatch')], text=True).strip()
            assert job.isdigit(), job
            report['jobs'].append({'rank': rank, 'phase': phase, 'job_id': job,
                                   'dependency': previous, 'gpus': 2 if phase == 'fit' else 1})
            path.write_text(json.dumps(report, indent=2) + '\n')
            previous = job
    report['status'] = 'submitted; completion and final collection pending'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
