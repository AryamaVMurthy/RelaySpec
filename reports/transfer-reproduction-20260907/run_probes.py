"""Four bounded operational checks; archive source remains unmodified."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
package=Path('/scratch/aryama.murthy/transfer-reproduction-20260907/package');os.chdir(package)
work=Path(os.environ['TRANSFER_WORK']);gpus=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(gpus)==4
out=Path('/home/aryama.murthy/relayspec-transfer-reproduction/outputs')/('probes-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True,exist_ok=False)
base=os.environ.copy();base.update(PYTHONPATH=str(package/'src'),PYTHONDONTWRITEBYTECODE='1',HF_HOME=str(work/'hf'),TMPDIR=str(work/'tmp'),XDG_CACHE_HOME=str(work/'cache_runtime'),VLLM_NO_USAGE_STATS='1',VLLM_USE_V2_MODEL_RUNNER='0',OMP_NUM_THREADS='8',VLLM_CONFIG_ROOT=str(work/'cache_runtime/config'))
for k in ['TRANSFER_CAPTURE','TRANSFER_MAPPED','TRANSFER_EAGER','VLLM_BATCH_INVARIANT']:base.pop(k,None)
tasks=[('generation',[sys.executable,'reproduce.py','generation-check','--gpus',gpus[0]]),('ar8',[sys.executable,'src/benchmark.py','ar8','--split','eval','--count','8','--cap','256','--tag','pretrain','--worker-index','0','--workers','1']),('native8',[sys.executable,'src/benchmark.py','native8','--split','eval','--count','8','--cap','256','--tag','pretrain','--worker-index','0','--workers','1']),('mapper',[sys.executable,'/home/aryama.murthy/relayspec-transfer-reproduction/probe_mapper_gpu.py'])]
def run(i):
 name,cmd=tasks[i];env=base.copy();env['CUDA_VISIBLE_DEVICES']=gpus[i];profile='eval' if i in [1,2] else 'training' if i==3 else 'generation'
 for k,sub in [('VLLM_CACHE_ROOT','vllm'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton')]:env[k]=str(work/'cache_runtime'/profile/sub)
 if profile=='eval':env['VLLM_BATCH_INVARIANT']='1'
 start=time.time()
 with (out/(name+'.log')).open('w') as f:
  p=subprocess.run(['timeout','--signal=TERM','--kill-after=10s','540s',*cmd],env=env,stdout=f,stderr=subprocess.STDOUT)
 r=dict(lane=i,name=name,command=cmd,exit_code=p.returncode,elapsed_seconds=time.time()-start,passed=p.returncode==0);(out/(name+'-status.json')).write_text(json.dumps(r,indent=2)+'\n');return r
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,range(4)))
(out/'results.json').write_text(json.dumps(results,indent=2)+'\n');assert all(r['passed'] for r in results),results
