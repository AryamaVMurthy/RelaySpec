"""Stage the exact bundled Math fidelity gate without changing shared environments."""
import hashlib,json,os,shutil
from pathlib import Path
import importlib.metadata as metadata
from huggingface_hub import snapshot_download

control=Path('/home/aryama.murthy/relayspec-auf-20260911/experiments/handoff_transfer')
root=Path('/scratch/aryama.murthy/handoff-transfer-20260911')
models=root/'models/qwen4';models.mkdir(parents=True,exist_ok=True)
old=Path('/scratch/aryama.murthy/transfer-reproduction-20260907/work/models/4b')
revisions={'target':'1cfa9a7208912126459214e8b04321603b3df60c','draft':'b74e3a329c4d963783143b1e970d95b002be72bd'}
for kind,revision in revisions.items():
    source=(old/kind).resolve();assert source.is_dir() and source.name==revision
    dest=models/kind
    if dest.exists():assert dest.resolve()==source
    else:dest.symlink_to(source,target_is_directory=True)
adapter=snapshot_download('witcheer/qwen3-4b-gsm8k-grpo',revision='f52e1ed1b3accd404d37a9621db0536dd99bdfc5',cache_dir=str(root/'hf'))
config=json.loads((Path(adapter)/'adapter_config.json').read_text())
assert not config.get('modules_to_save')
assert all('embed' not in x and 'lm_head' not in x for x in config['target_modules'])
work=root/'origin-math-gate';(work/'setup').mkdir(parents=True,exist_ok=True)
for split in ['train','eval']:
    shutil.copyfile(control/f'data/math_{split}.json',work/f'setup/{split}.json')
os.environ.update(AUF_WORKDIR=str(work),AUF_MODELS=str(models),AUF_DOMAIN='math',AUF_ADAPTER=adapter)
from common import load_upstream
load_upstream()
result={'models':str(models),'workdir':str(work),'adapter':adapter,'specforge':os.environ['SPECFORGE_ROOT'],
        'versions':{k:metadata.version(k) for k in ['torch','vllm','transformers','peft','accelerate']},
        'specforge_import':'passed','status':'prepared'}
assert result['versions']['peft']=='0.20.0'
(root/'prepared.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
