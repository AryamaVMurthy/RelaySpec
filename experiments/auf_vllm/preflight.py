"""Read-only environment/asset checks and a real CUDA operation in Slurm."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Preflight must run in a Slurm allocation")
    import torch
    import vllm
    import transformers
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Preflight requires exactly one visible GPU")
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    b = a @ a.T
    torch.cuda.synchronize()
    assert torch.isfinite(b).all()
    source = Path(os.environ["DFLASH_SOURCE"])
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    model_file = source/"dflash/model.py"
    sys.path.insert(0, str(source))
    from dflash.model import DFlashDraftModel
    root = Path(os.environ["TRANSFER_WORK"])
    models = {}
    for size in (4, 8):
        for kind in ("target", "draft"):
            model_dir = root/"models"/f"{size}b"/kind
            cfg = json.loads((model_dir/"config.json").read_text())
            models[f"{size}b-{kind}"] = {
                "path": str(model_dir), "resolved": str(model_dir.resolve()),
                "config": cfg, "safetensors": [x.name for x in model_dir.glob("*.safetensors")],
            }
    rollouts = sorted((root/"common/rollouts/train").glob("*.jsonl"))
    if not rollouts:
        raise RuntimeError("No archived training rollouts found")
    with rollouts[0].open() as stream:
        sample = json.loads(next(stream))
    result = {
        "status": "passed", "job_id": os.environ["SLURM_JOB_ID"], "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0), "cuda_test": "BF16 512-square GEMM finite",
        "python": sys.version, "torch": torch.__version__, "vllm": vllm.__version__,
        "transformers": transformers.__version__, "cuda": torch.version.cuda,
        "safetensors": importlib.metadata.version("safetensors"),
        "draft_source_commit": revision,
        "draft_source_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
        "draft_forward_signature": str(inspect.signature(DFlashDraftModel.forward)),
        "models": models, "rollout_shards": len(rollouts), "rollout_row_keys": list(sample),
        "sample_prompt_tokens": len(sample["prompt_token_ids"]), "sample_output_tokens": len(sample["output_ids"]),
        "elapsed_seconds": time.monotonic()-started,
    }
    (args.output/"preflight.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("status", "job_id", "gpu", "torch", "vllm", "transformers", "draft_forward_signature", "rollout_shards", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
