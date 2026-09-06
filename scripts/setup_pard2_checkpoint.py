"""Download and hash-check pinned PARD-2 artifacts on compute-node scratch."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

p = argparse.ArgumentParser()
p.add_argument("--config", type=Path, required=True)
a = p.parse_args()
c = json.loads(a.config.read_text())
paths = {}
for key, model in c["models"].items():
    paths[key] = snapshot_download(
        repo_id=model["id"],
        revision=model["revision"],
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
        local_files_only=key == "target",
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model"],
        max_workers=2,
    )
    for name, expected in model.get("weights_sha256", {}).items():
        with (Path(paths[key]) / name).open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"checkpoint hash mismatch: {name}")
Path(os.environ["RELAYSPEC_OUTPUT"], "setup-gate.json").write_text(
    json.dumps({"status": "pass", "models": c["models"], "paths": paths}, indent=2)
    + "\n"
)
