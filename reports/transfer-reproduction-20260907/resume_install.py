"""Complete the pinned installation, reusing verified downloads from bootstrap."""
import importlib.metadata as metadata
import json,os,subprocess,sys
from pathlib import Path
package=Path('/scratch/aryama.murthy/transfer-reproduction-20260907/package');os.chdir(package)
expected=json.loads((package/'provenance/environment.json').read_text())['packages'];env=os.environ.copy();env['PIP_CACHE_DIR']=str(package/'.cache/pip')
try:installed=metadata.version('vllm')
except metadata.PackageNotFoundError:installed=None
if installed!=expected['vllm']:
 wheel=json.loads((package/'provenance/vllm_wheel.json').read_text())[0]
 subprocess.run([sys.executable,'-m','pip','install','--no-deps',wheel['url']+'#'+wheel['digest'].replace(':','=')],env=env,check=True)
subprocess.run([sys.executable,'-m','pip','install','--extra-index-url','https://download.pytorch.org/whl/cu129','-r',str(package/'requirements.lock')],env=env,check=True)
subprocess.run([sys.executable,'-m','pip','check'],env=env,check=True)
subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_*.py'],env=env,check=True)
with (Path(os.environ['TRANSFER_WORK'])/'environment-check.json').open('w') as f:subprocess.run([sys.executable,'src/check_environment.py'],env=env,stdout=f,check=True)
