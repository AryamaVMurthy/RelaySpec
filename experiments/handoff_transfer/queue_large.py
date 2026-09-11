"""Complete the fixed-four-epoch large-data queue within two GPU slots."""
import json
import subprocess
from pathlib import Path


def main():
    control=Path('/home/aryama.murthy/relayspec-auf-20260911')
    code=control/'experiments/handoff_transfer'
    record=control/'outputs/handoff-transfer/large-fit-jobs.json'
    assert not record.exists(), 'Inspect existing job record before resubmission'
    record.parent.mkdir(parents=True,exist_ok=True)
    report={'fits':[{'records':4096,'updates':2048,'job_id':'31647'}],
            'evaluations':[],'epochs':4,'status':'submitting'}
    def save():record.write_text(json.dumps(report,indent=2)+'\n')
    def submit(args):
        job=subprocess.check_output(['sbatch','--parsable',*args],text=True).strip()
        assert job.isdigit(),job
        return job
    save();previous='31647'
    for count in (8192,16384,32768):
        dependency=previous+(':31645' if count==32768 else '')
        job=submit([f'--dependency=afterok:{dependency}',f'--export=ALL,RECORDS={count}',str(code/'large_fit.sbatch')])
        report['fits'].append({'records':count,'updates':count//2,'job_id':job})
        save();previous=job
    for fit in report['fits']:
        job=submit(['--array=0-2%2',f'--dependency=afterok:{previous}',
                    f"--export=ALL,RECORDS={fit['records']},UPDATES={fit['updates']},FIT_JOB={fit['job_id']},INITIALIZER_ROOT=/scratch/aryama.murthy/handoff-transfer-20260911/large-initializers",
                    str(code/'scaling_eval.sbatch')])
        report['evaluations'].append(dict(fit,evaluation_job_id=job))
        save();previous=job
    report['status']='submitted; verification and collection pending';save()
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
