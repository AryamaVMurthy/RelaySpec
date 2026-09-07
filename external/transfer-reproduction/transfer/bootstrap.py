"""Run with Python 3.12.13; install only inside this package's virtual environment."""
import sys,json,subprocess,os
from pathlib import Path
R=Path(__file__).resolve().parent
assert sys.version_info[:3]==(3,12,13),'Use Python 3.12.13, matching the recorded runtime'
venv=R/'.venv';subprocess.run([sys.executable,'-m','venv',str(venv)],check=True)
py=venv/'bin/python';env=os.environ.copy();env['PIP_CACHE_DIR']=str(R/'.cache/pip')
wheel=json.loads((R/'provenance/vllm_wheel.json').read_text())[0]
url=wheel['url']+'#'+wheel['digest'].replace(':','=')
subprocess.run([str(py),'-m','pip','install','--no-deps',url],env=env,check=True)
subprocess.run([str(py),'-m','pip','install','--extra-index-url','https://download.pytorch.org/whl/cu129','-r',str(R/'requirements.lock')],env=env,check=True)
subprocess.run([str(py),'-m','pip','check'],env=env,check=True)
subprocess.run([str(py),str(R/'src/check_environment.py')],env=env,check=True)
