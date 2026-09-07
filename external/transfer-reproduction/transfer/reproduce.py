"""Explicit stages; GPU stages capped at four child processes. No scheduler required."""
import argparse,os,sys,subprocess,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'src'))
from paths import WORK,put,code_hash
p=argparse.ArgumentParser();p.add_argument('stage',choices=['download','prepare','generation-check','approve-generation','generate','capture-check','approve-capture','capture','train','export','eval-check','approve-eval','evaluate','verify']);p.add_argument('--gpus',default='0,1,2,3');a=p.parse_args()
gpus=a.gpus.split(',');assert 1<=len(gpus)<=4 and len(set(gpus))==len(gpus)
base=os.environ.copy();base['PYTHONPATH']=str(R/'src');base['PYTHONDONTWRITEBYTECODE']='1'
base['HF_HOME']=str(WORK/'hf');base['TMPDIR']=str(WORK/'tmp');(WORK/'tmp').mkdir(parents=True,exist_ok=True)
base['XDG_CACHE_HOME']=str(WORK/'cache_runtime');base['VLLM_NO_USAGE_STATS']='1';base['VLLM_USE_V2_MODEL_RUNNER']='0';base['OMP_NUM_THREADS']='8'
base['VLLM_CONFIG_ROOT']=str(WORK/'cache_runtime/config')
for k in ['TRANSFER_CAPTURE','TRANSFER_MAPPED','TRANSFER_EAGER','VLLM_BATCH_INVARIANT']:base.pop(k,None)
def env(profile):
 e=base.copy()
 for key,sub in [('VLLM_CACHE_ROOT','vllm'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton')]:e[key]=str(WORK/'cache_runtime'/profile/sub)
 if profile=='capture':e['TRANSFER_CAPTURE']='1'
 if profile=='eval':e['VLLM_BATCH_INVARIANT']='1'
 return e
def run(script,*args,profile='data',gpu=None):
 e=env(profile)
 if gpu is not None:e['CUDA_VISIBLE_DEVICES']=gpu
 subprocess.run([sys.executable,str(R/'src'/script),*map(str,args)],env=e,check=True)
def parallel(tasks,profile):
 assert len(tasks)<=len(gpus)
 children=[]
 try:
  for gpu,(script,args) in zip(gpus,tasks):
   e=env(profile);e['CUDA_VISIBLE_DEVICES']=gpu
   children.append(subprocess.Popen([sys.executable,str(R/'src'/script),*map(str,args)],env=e))
  codes=[x.wait() for x in children];assert not any(codes),codes
 except BaseException:
  for x in children:
   if x.poll() is None:x.terminate()
  for x in children:x.wait()
  raise
if a.stage not in ['download','prepare','verify']:run('check_environment.py')
if a.stage=='download':run('download.py')
elif a.stage=='prepare':run('prepare.py')
elif a.stage=='generation-check':run('generate.py','--count',8,'--rank',0,'--world-size',2,'--probe',profile='generation',gpu=gpus[0])
elif a.stage=='approve-generation':
 # Operator runs this after reading the actual generated samples, not before.
 paths=list((WORK/'validation/probe_rollouts_r1/train').glob('*.jsonl'));assert paths
 rows=[json.loads(l) for q in paths for l in q.read_text().splitlines()]
 assert len(rows)==8 and all(0<len(x['output_ids'])<=4096 and x['full_ids']==x['prompt_token_ids']+x['output_ids'] for x in rows)
 put(WORK/'validation/generation_gate.json',{'passed':True,'manually_audited_by_operator':True,'code_hash':code_hash(),'source_hash':hashlib.sha256((R/'src/generate.py').read_bytes()).hexdigest()})
elif a.stage=='generate':
 assert json.loads((WORK/'validation/generation_gate.json').read_text())['code_hash']==code_hash()
 assert len(gpus)>=2,'Historical generation queue used two independent workers'
 parallel([('generate.py',['--count',16384,'--rank',i,'--world-size',2]) for i in range(2)],'generation')
 run('check_rollouts.py')
elif a.stage=='capture-check':
 run('capture.py','prepare',profile='capture')
 for mode in ['sample','dense']:
  tasks=[('capture.py',[mode,size]) for size in ['4','8']]
  if len(gpus)>=2:parallel(tasks,'capture')
  else:
   for script,args in tasks:run(script,*args,profile='capture',gpu=gpus[0])
 run('capture.py','audit',profile='capture')
elif a.stage=='approve-capture':
 c=json.loads((WORK/'cache/validation/comparison.json').read_text());assert c['passed']
 put(WORK/'cache/validation/gate.json',{'passed':True,'manually_audited_by_operator':True,'code_hash':code_hash()})
elif a.stage=='capture':
 assert json.loads((WORK/'cache/validation/gate.json').read_text())['code_hash']==code_hash()
 assert len(gpus)>=2
 parallel([('capture.py',['full',size]) for size in ['4','8']],'capture')
 run('check_features.py')
elif a.stage=='train':run('train.py',profile='training',gpu=gpus[0])
elif a.stage=='export':run('export.py',profile='training',gpu=gpus[0])
elif a.stage=='eval-check':
 for side in ['ar8','native8','mapped']:run('benchmark.py',side,'--split','eval','--count',8,'--cap',256,'--tag','check',profile='eval',gpu=gpus[0])
 run('verify.py','--tag','check','--count',8)
elif a.stage=='approve-eval':
 x=json.loads((WORK/'results/check.json').read_text());assert x['all_equal']
 put(WORK/'validation/eval_gate.json',{'passed':True,'manually_audited_by_operator':True,'code_hash':code_hash()})
elif a.stage=='evaluate':
 assert len(gpus)==4,'Use four visible GPUs for the recorded worker partition'
 assert json.loads((WORK/'validation/eval_gate.json').read_text())['passed'] and json.loads((WORK/'validation/eval_gate.json').read_text())['code_hash']==code_hash()
 parallel([('benchmark.py',['ar8','--split','eval','--count',128,'--cap',2048,'--tag','final','--worker-index',i,'--workers',4]) for i in range(4)],'eval')
 parallel([('benchmark.py',[side,'--split','eval','--count',128,'--cap',2048,'--tag','final','--worker-index',i,'--workers',2]) for side in ['native8','mapped'] for i in range(2)],'eval')
 run('verify.py')
elif a.stage=='verify':run('verify.py')
