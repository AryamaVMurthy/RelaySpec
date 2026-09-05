"""CPU setup of pinned public PARD weights and its isolated inference overlay."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

from relayspec.pard_adapter import PARD_SOURCE_SHA256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if (
        hashlib.sha256(Path("vendor/pard/pard/pard_infer.py").read_bytes()).hexdigest()
        != PARD_SOURCE_SHA256
    ):
        raise ValueError("unreviewed public PARD source")
    paths = {}
    for key in ("target", "proposer"):
        paths[key] = snapshot_download(
            repo_id=config[key]["id"],
            revision=config[key]["revision"],
            cache_dir=os.environ["TRANSFORMERS_CACHE"],
            local_files_only=key == "target",
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
            max_workers=2,
        )
    weights = Path(paths["proposer"]) / "model.safetensors"
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != config["proposer"]["weights_sha256"]:
        raise ValueError("released PARD weights differ from pinned LFS blob")
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "tokenizers",
            "huggingface-hub",
            "numpy",
            "loguru",
            "click",
        )
    }
    if versions["transformers"] != "4.51.3":
        raise ValueError("PARD inference overlay has the wrong Transformers version")
    (Path(os.environ["RELAYSPEC_OUTPUT"]) / "setup-gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "overlay": os.environ["PARD_OVERLAY"],
                "models": paths,
                "versions": versions,
                "weights_sha256": digest.hexdigest(),
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "scope": "CPU-only dependency/checkpoint preparation. No GPU inference gate or speed claim.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
