"""Idempotent batch8 scheduling on node07's four physical GPUs.

Fits are independent one-GPU jobs. Each family first creates its shared
AR/original/ZIP references, then remaining evaluations can run in parallel.
"""
import json,shlex,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent
remote='/home/aryama.murthy/relayspec-auf-20260911/experiments/handoff_transfer/exact_scope'
ledger=p/'jobs.json';jobs=json.loads(ledger.read_text()) if ledger.exists() else []

def submit(script,name,env,dep=None,gpus=1,array=None,nice=None):
    existing=[j for j in jobs if j['name']==name]
    if existing:
        assert len(existing)==1 and existing[0]['env']==env
        return existing[0]['job']
    args=['sbatch','--parsable','--job-name',name,'--nodelist=node07','--export','ALL,'+','.join(f'{k}={v}' for k,v in env.items())]
    if dep:args+=['--dependency','afterok:'+':'.join(map(str,dep if isinstance(dep,list) else [dep]))]
    if gpus:args+=['--gres',f'gpu:{gpus}']
    if array:args+=['--array',array]
    if nice is not None:args+=['--nice',str(nice)]
    args+=[remote+'/'+script]
    job=int(subprocess.check_output(['ssh','turing',shlex.join(args)],text=True).strip().split(';')[0])
    jobs.append(dict(job=job,name=name,node='node07',env=env,dependency=dep,script=script,gpus=gpus,
                     **({'array':array} if array else {}),**({'nice':nice} if nice is not None else {})))
    ledger.write_text(json.dumps(jobs,indent=2)+'\n');print(job,name,flush=True)
    return job

def token_tasks(family,dep=None):
    result=[]
    for task in json.loads((p/f'{family}-tasks.json').read_text()):
        env=dict(FAMILY=family,KIND=task['kind'],MATRIX_OBJECTIVE=task['objective'],MATRIX_LR=task['lr'],TRAIN_STEPS=512)
        if family=='cross':env['CROSS_INDEX']='/scratch/aryama.murthy/handoff-transfer-20260911/cross-full4096/index.json'
        job=submit('cross_fit.sbatch' if family=='cross' else 'fit.sbatch',f'b8-{family}-{task["kind"]}-{task["objective"]}',env,dep)
        result.append((job,env))
    return result

def features(family,dep=None):
    result=[]
    for objective in ['feature_ce','forward_kl','reverse_kl']:
        env=dict(FAMILY=family,KIND='feature',MATRIX_OBJECTIVE=objective,MATRIX_LR=.001)
        job=submit('feature_fit.sbatch',f'b8-feature-{family}-{objective}',env,dep)
        result.append((job,env))
    return result

def evaluate(cases,first=None):
    reference=first
    for fit,env in cases:
        family=env['FAMILY'];name=f'b8-eval-{family}-{env["KIND"]}-{env["MATRIX_OBJECTIVE"]}'
        job=submit('eval.sbatch',name,env,[fit,reference] if reference else fit)
        if reference is None:reference=job
    return reference

# Start the first full-batch evaluation immediately while more fits fill the GPUs.
first=next(j for j in jobs if j['name']=='b8-q8-five_maps-ce')
q_first=evaluate([(first['job'],first['env'])])
q=token_tasks('q8')+features('q8')
l=token_tasks('llama')
paired=submit('assets_ready.sbatch','b8-llama-paired-ready',{},gpus=0)
l+=features('llama',paired)
# Prefer ready evaluations, then fill free node07 GPUs with independent capture
# chunks. Pinning every job to this four-GPU node enforces the study ceiling.
data=submit('../cross_data_array.sbatch','b8-cross-data',{},array='0-63%4',nice=100)
assemble=submit('../cross_assemble.sbatch','b8-cross-assemble',{},data,gpus=0)
init=submit('cross_initializers.sbatch','b8-cross-initializers',{'CROSS_INDEX':'/scratch/aryama.murthy/handoff-transfer-20260911/cross-full4096/index.json'},assemble)
x=token_tasks('cross',init)+features('cross',init)
evaluate([case for case in q if case[0]!=first['job']],q_first)
evaluate(l);evaluate(x)
# Two method cross-checks plus one shared AR reference, after the primary matrix.
all_evals=[j['job'] for j in jobs if j['script']=='eval.sbatch']
references=submit('gpu_references.sbatch','b8-gpu-matched-references',{},all_evals,gpus=4)
for mode in ['ar','ce','auf']:
    submit('transformers.sbatch',f'b8-transformers-q8-{mode}',{'MODE':mode},references)
submit('collect.sbatch','b8-final-audit',{},[j['job'] for j in jobs if j['script']=='transformers.sbatch'],gpus=0)
