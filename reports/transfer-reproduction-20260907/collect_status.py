"""Read-only bounded collection of transfer reproduction status and small artifacts."""
import json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
for _ in range(360):
 ledger=json.loads((root/'jobs.json').read_text());ids=','.join(str(x['id']) for x in ledger['jobs'])
 p=subprocess.run(['ssh','-o','BatchMode=yes','turing',f'sacct -X -P -n -j {ids} --format=JobID,State,Elapsed,AllocTRES'],text=True,capture_output=True,timeout=30)
 (root/'slurm-status.txt').write_text(p.stdout if p.returncode==0 else p.stderr)
 if p.returncode==0:
  for remote,local in [('/home/aryama.murthy/relayspec-transfer-reproduction/logs/','logs/'),('/home/aryama.murthy/relayspec-transfer-reproduction/outputs/','outputs/')]:
   dest=root/local;dest.mkdir(exist_ok=True)
   r=subprocess.run(['rsync','-az','--ignore-missing-args','--exclude','*.pt','--exclude','*.safetensors','turing:'+remote,str(dest)+'/'],capture_output=True,text=True,timeout=45)
   if r.returncode not in [0,23]:raise RuntimeError(r.stderr)
 (root/'collector-status.json').write_text(json.dumps(dict(updated_unix=time.time(),read_only=True,iterations_limit=360,interval_seconds=60))+'\n')
 time.sleep(60)
