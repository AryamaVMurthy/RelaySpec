"""Collect the fixed campaign without resubmitting or changing any experiment."""
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/home/aryama.murthy/native-joint-20260908-v48'
jobs=json.loads((ROOT/'reports/confirmation-jobs.json').read_text())
ids=','.join(r['job'] for r in jobs)
raw=subprocess.check_output(['ssh','turing',f'sacct -j {ids} --noheader --parsable2 --format=JobIDRaw,State,ExitCode,Elapsed'],text=True)
states={line.split('|')[0]:line.split('|')[1:] for line in raw.splitlines() if '.' not in line.split('|')[0]}
summary=[]
for item in jobs:
    job=item['job'];state=states.get(job,['UNKNOWN'])[0]
    row=dict(item,state=state,accounting=states.get(job));summary.append(row)
    if state in {'PENDING','UNKNOWN'} or state.startswith('CANCELLED'):continue
    destination=ROOT/f'reports/run-{job}'
    destination.mkdir(exist_ok=True)
    subprocess.run(['rsync','-az','--exclude=*.pt',f'turing:{REMOTE}/outputs/{job}/',str(destination)+'/'],check=True)
    metadata=destination/'metadata';metadata.mkdir(exist_ok=True)
    subprocess.run(['rsync','-az',f'turing:{REMOTE}/metadata/{job}*',str(metadata)+'/'],check=True)
    (metadata/'sacct.txt').write_text('|'.join([job]+states[job])+'\n')
(ROOT/'reports/confirmation-status.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
