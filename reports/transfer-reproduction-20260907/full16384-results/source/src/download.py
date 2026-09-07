import json,shutil,hashlib
from huggingface_hub import snapshot_download,hf_hub_download
from paths import PACKAGE,WORK,put
pins=json.loads((PACKAGE/'pins.json').read_text())
for key,c in pins.items():
 if key=='dataset':continue
 snapshot_download(c['repo'],revision=c['revision'],local_dir=WORK/c['path'],allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.tiktoken','LICENSE*','README*','*.py'])
c=pins['dataset'];p=hf_hub_download(c['repo'],c['file'],repo_type='dataset',revision=c['revision'],cache_dir=WORK/'download_cache')
out=WORK/'dataset/train-00000.parquet';out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out)
put(WORK/'download_provenance.json',{'pins':pins,'dataset_sha256':hashlib.sha256(out.read_bytes()).hexdigest()})
