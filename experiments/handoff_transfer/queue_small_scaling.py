"""Submit the remaining small-data fits/evaluations as bounded serial lanes."""
import json
import subprocess
from pathlib import Path


def main():
    control = Path('/home/aryama.murthy/relayspec-auf-20260911')
    code = control / 'experiments/handoff_transfer'
    record = control / 'outputs/handoff-transfer/small-scaling-jobs.json'
    assert not record.exists(), 'Already submitted; inspect recorded job IDs instead of resubmitting'
    record.parent.mkdir(parents=True, exist_ok=True)
    jobs = {'fits': [], 'evaluations': [], 'status': 'submitting'}

    def submit(arguments):
        job = subprocess.check_output(['sbatch', '--parsable', *arguments], text=True).strip()
        assert job.isdigit(), job
        return job

    def save():
        record.write_text(json.dumps(jobs, indent=2) + '\n')

    save()
    # These prerequisites finish the N16 end-to-end gate and both compute pools.
    previous = '31610:31612:31613'
    for count in (32, 64, 128, 256, 512, 1024, 2048, 4096):
        job = submit([f'--dependency=afterok:{previous}',
                      f'--export=ALL,RECORDS={count},UPDATES=1024', str(code / 'scaling_fit.sbatch')])
        jobs['fits'].append({'records': count, 'job_id': job, 'gpus': 2})
        save()
        previous = job
    # Each array is at most two GPUs; arrays run serially after all fits.
    # The independent dense-capture lane may use at most two additional GPUs.
    for fit in jobs['fits']:
        count = fit['records']
        job = submit(['--array=0-2%2', f'--dependency=afterok:{previous}',
                      f"--export=ALL,FIT_JOB={fit['job_id']},RECORDS={count},UPDATES=1024",
                      str(code / 'scaling_eval.sbatch')])
        jobs['evaluations'].append({'records': count, 'fit_job': fit['job_id'], 'job_id': job,
                                   'array': '0-2%2', 'requests': 128, 'cap': 2048, 'repetitions': 3})
        save()
        previous = job
    jobs['status'] = 'submitted; completion and collection pending'
    save()
    print(json.dumps(jobs, indent=2), flush=True)


if __name__ == '__main__':
    main()
