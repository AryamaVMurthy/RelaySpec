"""Download only the pinned original Qwen14 target, with provenance."""
import json
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from .pilot_data import write,sha

root=Path("/scratch/aryama.murthy/relayspec-auf-20260911")
assert shutil.disk_usage(root).free > 80*1024**3
revision="40c069824f4251a91eefaf281ebe4c544efd3e18"
path=Path(snapshot_download(repo_id="Qwen/Qwen3-14B",revision=revision,
    local_dir=root/"models/qwen14-target",cache_dir=root/"download_cache",
    allow_patterns=["*.json","*.safetensors","*.txt","*.model"],max_workers=2))
config=json.loads((path/"config.json").read_text())
assert config["hidden_size"] == 5120 and config["num_hidden_layers"] == 40
write(root/"models/qwen14-target-provenance.json",{"repo_id":"Qwen/Qwen3-14B","revision":revision,
    "config_sha256":sha(path/"config.json"),"target_adapters":None,
    "weight_files":[{"name":p.name,"bytes":p.stat().st_size} for p in sorted(path.glob("*.safetensors"))]})
print("Pinned Qwen3-14B download complete",flush=True)
