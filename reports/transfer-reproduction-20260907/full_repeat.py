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
# Prespecified repeated comparison with method-to-GPU assignment reversed.
assert len(gpus)==4
with concurrent.futures.ThreadPoolExecutor(4) as pool:
 fs=[]
 for mode in ['native8','mapped']:
  for i in range(2):
   gpu=gpus[2+i] if mode=='native8' else gpus[i]
   fs.append(pool.submit(run,mode+'-repeat1-'+str(i),['src/benchmark.py',mode,'--count','128','--cap','2048','--tag','full16384-final','--repeat','1','--worker-index',str(i),'--workers','2'],'eval',gpu,1200))
 for f in fs:f.result()
rows={}
for mode in ['ar8','native8','mapped']:
 repeat=0 if mode=='ar8' else 1
 rr=[json.loads(l) for f in sorted((work/'measurements/full16384-final').glob('worker_*/'+mode+'-r'+str(repeat)+'.jsonl')) for l in f.read_text().splitlines()]
 assert len(rr)==128 and all(x['timing_valid'] for x in rr)
 rows[mode]={x['group_id']:x for x in rr};assert len(rows[mode])==128
keys=rows['ar8'].keys();assert all(v.keys()==keys for v in rows.values())
for k in keys:
 a,b,c=[rows[m][k] for m in ['ar8','native8','mapped']]
 assert a['output_ids']==b['output_ids']==c['output_ids']
 assert a['prompt_ids']==b['prompt_ids']==c['prompt_ids']
 assert a['finish_reason']==b['finish_reason']==c['finish_reason']
(out/'repeat_gate.json').write_text(json.dumps({'passed':True,'count':128,'cap':2048,'repeat':1,'assignment':'native GPU2/3, mapped GPU0/1; opposite first run'}))
