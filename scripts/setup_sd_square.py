"""Prepare and verify an isolated public SD-square environment on a CPU allocation."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    overlay = Path(os.environ["SD_SQUARE_OVERLAY"])
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    if args.install:
        uv = str(Path.home() / ".local/bin/uv")
        common = [
            uv,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            str(overlay),
        ]
        # Preserve the verified base PyTorch/CUDA build. The three Lightning
        # roots would otherwise pull a new torch wheel into the overlay.
        roots = {
            name: config["runtime"][name]
            for name in ("lightning", "pytorch-lightning", "torchmetrics")
        }
        subprocess.run(
            [*common, "--no-deps", *[f"{n}=={v}" for n, v in roots.items()]], check=True
        )
        from packaging.requirements import Requirement

        dependencies = []
        for dist in importlib.metadata.distributions(path=[str(overlay)]):
            for raw in dist.requires or []:
                item = Requirement(raw)
                if item.name.lower().replace("_", "-") in {*roots, "torch"}:
                    continue
                if item.marker is None or item.marker.evaluate({"extra": ""}):
                    dependencies.append(str(item))
        packages = [f"{n}=={v}" for n, v in config["runtime"].items() if n not in roots]
        subprocess.run(
            [
                *common,
                *sorted(set(dependencies)),
                *packages,
                "jsonargparse[signatures]==4.38.0",
            ],
            check=True,
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(overlay)
        subprocess.run(
            [sys.executable, __file__, "--config", str(args.config)],
            env=env,
            check=True,
        )
        return
    gate = {
        "status": "fail",
        "phase": "source_and_import",
        "overlay": str(overlay),
        "config_sha256": digest(args.config),
        "source_commit": config["source_commit"],
    }
    try:
        source = Path("vendor/sd-square").resolve()
        actual = {
            str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*.py"))
        }
        if actual != config["source_sha256"]:
            raise ValueError("public SD-square sources differ from the declaration")
        gate["source_sha256"] = actual
        gate["versions"] = {
            n: importlib.metadata.version(n)
            for n in (*config["runtime"], "torch", "tokenizers", "huggingface-hub")
        }
        if any(gate["versions"][n] != v for n, v in config["runtime"].items()):
            raise ValueError("SD-square package versions differ from the declaration")
        if gate["versions"]["torch"].split("+")[0] != "2.9.1":
            raise ValueError("SD-square setup replaced the verified base PyTorch")
        sys.path.insert(0, str(source))
        import main as upstream

        gate["training_class"] = upstream.TrainingModule.__name__
        gate["phase"] = "checkpoint_preparation"
        from huggingface_hub import snapshot_download

        gate["models"] = {}
        for name in ("target", "drafter"):
            gate["models"][name] = snapshot_download(
                repo_id=config[name]["id"],
                revision=config[name]["revision"],
                cache_dir=os.environ["TRANSFORMERS_CACHE"],
                local_files_only=name == "target",
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
                max_workers=2,
            )
        weights = Path(gate["models"]["drafter"]) / "model.safetensors"
        gate["drafter_weights_sha256"] = digest(weights)
        if gate["drafter_weights_sha256"] != config["drafter"]["weights_sha256"]:
            raise ValueError("base drafter weights differ from the pinned LFS blob")
        gate.update(
            status="pass",
            phase="complete",
            python=sys.version,
            scope="CPU source/import/checkpoint preparation only. Real frozen training, "
            "gradient checks, checkpoint reload and decoding require a GPU pilot.",
        )
    except Exception:
        gate["error"] = traceback.format_exc()
        raise
    finally:
        (output / "setup-gate.json").write_text(json.dumps(gate, indent=2) + "\n")


if __name__ == "__main__":
    main()
