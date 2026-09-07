import os,sys,json,subprocess,time,concurrent.futures
from pathlib import Path
root=Path('/scratch/aryama.murthy/transfer-reproduction-20260907'); package=root/'quick512-package';work=Path('/scratch/aryama.murthy/rs-q512');os.chdir(package)
gpus=os.environ['CUDA_VISIBLE_DEVICES'].split(',');out=Path('/home/aryama.murthy/relayspec-transfer-reproduction/outputs')/('quick-'+os.environ['SLURM_JOB_ID']);out.mkdir()
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
if sys.argv[1]=='captureonly':
 run('capture-check',['reproduce.py','capture-check','--gpus',','.join(gpus[:2])],'capture',','.join(gpus[:2]),900)
elif sys.argv[1]=='check':
 def capturecheck():run('capture-check',['reproduce.py','capture-check','--gpus',','.join(gpus[:2])],'capture',','.join(gpus[:2]),900)
 with concurrent.futures.ThreadPoolExecutor(3) as pool:
  fs=[pool.submit(capturecheck)]
  for mode,gpu in zip(['ar8','native8'],gpus[2:]):fs.append(pool.submit(run,mode,['src/benchmark.py',mode,'--count','8','--cap','512','--tag','quick512'],'eval',gpu))
  for f in fs:f.result()
else:
 run('approve-capture',['reproduce.py','approve-capture'],'capture',gpus[0])
 run('capture-full',['reproduce.py','capture','--gpus',','.join(gpus[:2])],'capture',','.join(gpus[:2]),900)
 run('train',['reproduce.py','train','--gpus',gpus[0]],'training',gpus[0])
 run('export',['reproduce.py','export','--gpus',gpus[0]],'training',gpus[0])
 run('mapped',['src/benchmark.py','mapped','--count','8','--cap','512','--tag','quick512'],'eval',gpus[0])
