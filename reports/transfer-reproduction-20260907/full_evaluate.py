import os,sys,json,subprocess,time,concurrent.futures
from pathlib import Path
root=Path('/scratch/aryama.murthy/transfer-reproduction-20260907'); package=root/'full-package';work=root/'work';os.chdir(package)
gpus=os.environ['CUDA_VISIBLE_DEVICES'].split(',');out=Path('/home/aryama.murthy/relayspec-transfer-reproduction/outputs')/('full-'+os.environ['SLURM_JOB_ID']);out.mkdir()
base=os.environ.copy();base.update(TRANSFER_WORK=str(work),PYTHONPATH=str(package/'src'),PYTHONDONTWRITEBYTECODE='1',HF_HOME=str(work/'hf'),TMPDIR=str(work/'tmp'),VLLM_NO_USAGE_STATS='1',VLLM_USE_V2_MODEL_RUNNER='0',OMP_NUM_THREADS='8',VLLM_CONFIG_ROOT=str(work/'cache_runtime/config'))
for k in ['TRANSFER_CAPTURE','TRANSFER_MAPPED','TRANSFER_EAGER','VLLM_BATCH_INVARIANT']:base.pop(k,None)
def run(name,args,profile,gpu,timeout=540):
 e=base.copy();e['CUDA_VISIBLE_DEVICES']=gpu
 for k,s in [('VLLM_CACHE_ROOT','vllm'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton')]:e[k]=str(root/'work/cache_runtime'/profile/s)
 if profile=='eval':e['VLLM_BATCH_INVARIANT']='1'
 if profile=='capture':e['TRANSFER_CAPTURE']='1'
 t=time.time()
 with (out/(name+'.log')).open('w') as f:r=subprocess.run(['timeout','--signal=TERM','--kill-after=10s',str(timeout)+'s',sys.executable,*args],env=e,stdout=f,stderr=subprocess.STDOUT)
 status={'name':name,'exit_code':r.returncode,'seconds':time.time()-t};(out/(name+'.status.json')).write_text(json.dumps(status));assert r.returncode==0,status
# The eight-prompt gate compares the trained full-data mapper to the completed matched pilot baselines.
paths={'mapped':work/'measurements/full16384-quick/worker_0/mapped-r0.jsonl'}
for m in ['ar8','native8']:paths[m]=Path('/scratch/aryama.murthy/rs-q512/measurements/quick512/worker_0')/(m+'-r0.jsonl')
rows={m:[json.loads(l) for l in p.read_text().splitlines()] for m,p in paths.items()}
assert all(len(v)==8 for v in rows.values())
for a,b,c in zip(rows['ar8'],rows['native8'],rows['mapped']):
 assert a['group_id']==b['group_id']==c['group_id']
 assert a['prompt_ids']==b['prompt_ids']==c['prompt_ids']
 assert a['output_ids']==b['output_ids']==c['output_ids']
 assert a['finish_reason']==b['finish_reason']==c['finish_reason']
 assert all(x['timing_valid'] for x in [a,b,c])
(out/'eight_prompt_gate.json').write_text(json.dumps({'passed':True,'count':8,'cap':512,'sources':{m:str(p) for m,p in paths.items()}}))
assert len(gpus)==4
with concurrent.futures.ThreadPoolExecutor(4) as pool:
 fs=[pool.submit(run,'ar8-'+str(i),['src/benchmark.py','ar8','--count','128','--cap','2048','--tag','full16384-final','--worker-index',str(i),'--workers','4'],'eval',gpus[i],2400) for i in range(4)]
 for f in fs:f.result()
with concurrent.futures.ThreadPoolExecutor(4) as pool:
 fs=[]
 for j,mode in enumerate(['native8','mapped']):
  for i in range(2):fs.append(pool.submit(run,mode+'-'+str(i),['src/benchmark.py',mode,'--count','128','--cap','2048','--tag','full16384-final','--worker-index',str(i),'--workers','2'],'eval',gpus[2*j+i],1200))
 for f in fs:f.result()
# Distinguish cross-method correctness from historical token equality for this new-data realization.
import hashlib
by={};source={}
for mode in ['ar8','native8','mapped']:
 files=sorted((work/'measurements/full16384-final').glob('worker_*/'+mode+'-r0.jsonl'))
 rr=[json.loads(l) for p in files for l in p.read_text().splitlines()]
 assert len(rr)==128 and all(x['timing_valid'] and x['output_tokens']==len(x['output_ids']) for x in rr)
 by[mode]={x['group_id']:x for x in rr};assert len(by[mode])==128
 source[mode]=[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
expected={x['group_id']:x for x in map(json.loads,(package/'reference/ar8.jsonl').read_text().splitlines())}
keys=by['ar8'].keys();assert all(v.keys()==keys for v in by.values())
diff=[];hist=[]
for k in keys:
 a,b,c=[by[m][k] for m in ['ar8','native8','mapped']]
 assert a['prompt_ids']==b['prompt_ids']==c['prompt_ids']==expected[k]['prompt_ids']
 if not(a['output_ids']==b['output_ids']==c['output_ids'] and a['finish_reason']==b['finish_reason']==c['finish_reason']):diff.append(k)
 if any(x['output_ids']!=expected[k]['output_ids'] for x in [a,b,c]):hist.append(k)
stats={}
for m,rr in by.items():
 n=sum(x['output_tokens'] for x in rr.values());s=sum(x['wall_seconds'] for x in rr.values());stats[m]={'tokens':n,'seconds':s,'tps':n/s,'length_stops':sum(x['finish_reason']=='length' for x in rr.values())}
result={'count':128,'cap':2048,'training_records':16384,'epochs':3,'all_equal':not diff,'different_groups':diff,'different_from_historical_output':hist,'stats':stats,'sources':source,'historical_training_rollouts_match':False}
(work/'results').mkdir(exist_ok=True)
(work/'results/full16384-final.json').write_text(json.dumps(result,indent=2))
(out/'comparison.json').write_text(json.dumps(result,indent=2))
assert not diff,'Methods differ: preserve and investigate before performance claims'
