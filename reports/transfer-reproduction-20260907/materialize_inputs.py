"""Reuse exact pinned HF snapshots; obtain the archive's pinned dataset shard."""
import hashlib
import json
import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

package=Path('/scratch/aryama.murthy/transfer-reproduction-20260907/package')
work=Path(os.environ['TRANSFER_WORK']);pins=json.loads((package/'pins.json').read_text())
old_cache=Path('/scratch/aryama.murthy/factorspec-runtime-20260826/huggingface/hub')
records={}
def weight_record(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 value=h.hexdigest();blob=path.resolve().name
 if len(blob)==64 and all(c in '0123456789abcdef' for c in blob):assert value==blob, path
 return dict(name=path.name,bytes=path.stat().st_size,sha256=value)
for name,pin in pins.items():
 if name=='dataset':continue
 snapshot=Path(snapshot_download(pin['repo'],revision=pin['revision'],cache_dir=old_cache,local_files_only=True))
 assert snapshot.name==pin['revision']
 files=list(snapshot.glob('*.safetensors'));assert files and all(p.stat().st_size>0 for p in files)
 index=snapshot/'model.safetensors.index.json'
 if index.exists():assert all((snapshot/f).is_file() for f in set(json.loads(index.read_text())['weight_map'].values()))
 target=work/pin['path'];target.parent.mkdir(parents=True,exist_ok=True)
 if target.is_symlink():assert target.resolve()==snapshot.resolve()
 else:target.symlink_to(snapshot,target_is_directory=True)
 records[name]=dict(repo=pin['repo'],revision=pin['revision'],snapshot=str(snapshot),config_sha256=hashlib.sha256((snapshot/'config.json').read_bytes()).hexdigest(),weight_files=[weight_record(f) for f in sorted(files)])
pin=pins['dataset'];source=Path(hf_hub_download(pin['repo'],pin['file'],repo_type='dataset',revision=pin['revision'],cache_dir=work/'download_cache'))
target=work/'dataset/train-00000.parquet';target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
hash=hashlib.sha256(target.read_bytes()).hexdigest()
(work/'download_provenance.json').write_text(json.dumps(dict(pins=pins,dataset_sha256=hash,model_materialization=records,scope='Exact pinned snapshots reused from existing HF cache. No prior mapper weights, generated rollouts, or feature tensors reused.'),indent=2)+'\n')
print(json.dumps(dict(status='pass',dataset_sha256=hash,models=records)),flush=True)
