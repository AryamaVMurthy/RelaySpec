"""Submit two serial lanes, capped at two GPUs per lane. Run locally."""
import json,shlex,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent
remote='/home/aryama.murthy/relayspec-auf-20260911/experiments/handoff_transfer/exact_scope'
ledger=p/'jobs.json'
if ledger.exists():raise RuntimeError('Submission ledger exists; do not duplicate live jobs')
jobs=[]
def submit(script,name,node,env,dep=None,gpus=None):
    args=['sbatch','--parsable','--job-name',name,'--nodelist',node,'--export','ALL,'+','.join(f'{k}={v}' for k,v in env.items())]
    if dep:args+=['--dependency','afterok:'+str(dep)]
    if gpus:args+=['--gres',f'gpu:{gpus}']
    args+=[remote+'/'+script]
    result=subprocess.check_output(['ssh','turing',shlex.join(args)],text=True).strip()
    job=int(result.split(';')[0]);jobs.append(dict(job=job,name=name,node=node,env=env,dependency=dep,script=script));ledger.write_text(json.dumps(jobs,indent=2)+'\n');print(job,name,flush=True)
    return job

def lane(family,node,dep=None):
    for task in json.loads((p/f'{family}-tasks.json').read_text()):
        env=dict(FAMILY=family,KIND=task['kind'],MATRIX_OBJECTIVE=task['objective'],MATRIX_LR=task['lr'],TRAIN_STEPS=2000)
        if family=='cross':env['CROSS_INDEX']='/scratch/aryama.murthy/handoff-transfer-20260911/cross-full4096/index.json'
        dep=submit('cross_fit.sbatch' if family=='cross' else 'fit.sbatch',f'exact32-{family}-{task["kind"]}-{task["objective"]}',node,env,dep,2)
        dep=submit('eval.sbatch',f'exact128-{family}-{task["kind"]}-{task["objective"]}',node,env,dep,1)
    for objective in ['feature_ce','forward_kl','reverse_kl']:
        env=dict(FAMILY=family,KIND='feature',MATRIX_OBJECTIVE=objective,MATRIX_LR=.001)
        dep=submit('feature_fit.sbatch',f'exact-feature-{family}-{objective}',node,env,dep,1)
        dep=submit('eval.sbatch',f'exact128-{family}-{objective}',node,env,dep,1)
    return dep
q=lane('q8','node07');l=lane('llama','node06')
subprocess.run(['ssh','turing',f'scontrol update JobId=31806 Dependency=afterok:{q} ArrayTaskThrottle=2'],check=True)
subprocess.run(['ssh','turing','scontrol release 31806'],check=True)
lane('cross','node07',31808)
