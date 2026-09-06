"""One bounded, single-GPU research screen inside a four-GPU allocation."""

import argparse
import copy
import gc
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
import yaml

from relayspec.ar_paper_evidence import summarize


def prepare(spec, root):
    source = Path(spec["checkpoint"])
    base = torch.load(source, map_location="cpu", weights_only=True)
    weight = base["relay"]["projection.weight"]
    provenance = {
        "parent_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "shape": list(weight.shape),
        "transform": spec["transform"],
    }
    variants = {"relay_base": str(source)}

    def save(name, checkpoint):
        path = root / (name + ".pt")
        torch.save(checkpoint, path)
        variants[name] = str(path)

    if spec["transform"] == "svd":
        torch.manual_seed(1729)
        w = weight.float().cuda()
        u, s, v = torch.svd_lowrank(
            w, q=min(max(spec["ranks"]) + 32, min(w.shape)), niter=4
        )
        provenance["randomized_svd"] = {
            "seed": 1729,
            "power_iterations": 4,
            "q": len(s),
        }
        provenance["relative_weight_errors"] = {}
        for rank in spec["ranks"]:
            cp = copy.deepcopy(base)
            cp["factorized_rank"] = rank
            cp["relay"] = {
                "projection.0.weight": v[:, :rank].T.contiguous().cpu(),
                "projection.1.weight": (u[:, :rank] * s[:rank]).cpu(),
            }
            provenance["relative_weight_errors"][str(rank)] = float(
                (w - (u[:, :rank] * s[:rank]) @ v[:, :rank].T).norm() / w.norm()
            )
            save(f"relay_svd{rank}", cp)
    elif spec["transform"] == "drop_taps":
        taps = base["target_layer_ids"]
        width = weight.shape[1] // len(taps)
        provenance["target_layer_ids"] = list(taps)
        provenance["interpretation"] = (
            "Zero weight blocks; all taps still captured and input normalization unchanged. Causal feature-use ablation, not a memory-saving implementation."
        )
        for i, layer in enumerate(taps):
            cp = copy.deepcopy(base)
            cp["relay"]["projection.weight"][:, i * width : (i + 1) * width] = 0
            save(f"relay_drop{layer}", cp)
    elif spec["transform"] == "data_interpolation":
        small_path = Path(spec["small_checkpoint"])
        small = torch.load(small_path, map_location="cpu", weights_only=True)
        assert small["target_layer_ids"] == base["target_layer_ids"]
        provenance["small_sha256"] = hashlib.sha256(small_path.read_bytes()).hexdigest()
        variants["relay_small"] = str(small_path)
        for alpha in [0.25, 0.5, 0.75]:
            cp = copy.deepcopy(base)
            cp["relay"]["projection.weight"] = weight * alpha + small["relay"][
                "projection.weight"
            ] * (1 - alpha)
            save(f"relay_blend{int(alpha * 100)}", cp)
    else:
        raise ValueError("Unknown research transform")
    (root / "transform.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return variants


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    spec = json.loads(a.spec.read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    variants = prepare(spec, a.output)
    gc.collect()
    torch.cuda.empty_cache()
    config = yaml.safe_load(Path(spec["config"]).read_text())
    family = config["proposer"]["family"]
    config["resources"]["gpu_count"] = 1
    config["benchmark"].update(
        max_prompts=8,
        research_single_gpu=True,
        methods=[
            "native_ar",
            "native_target_dflash" if family == "dflash" else "native_target_eagle3",
            *variants,
        ],
    )
    config["benchmark"].pop("isolate_methods", None)
    if family == "dflash":
        config["benchmark"]["unload_source_trunk"] = True
    config["generation"]["max_new_tokens"] = 128
    config["relay_probe"]["variants"] = variants
    cfg = a.output / "config.yaml"
    cfg.write_text(yaml.safe_dump(config, sort_keys=False))
    py = os.environ["RELAYSPEC_PYTHON"]
    env = {**os.environ, "RELAYSPEC_OUTPUT": str(a.output)}
    subprocess.run(
        [
            py,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=1",
            "scripts/benchmark_relay.py"
            if family == "dflash"
            else "scripts/benchmark_eagle3.py",
            "--config",
            str(cfg),
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [
            py,
            "scripts/check_run_complete.py",
            "--config",
            str(cfg),
            "--output",
            str(a.output),
        ],
        check=True,
    )
    rows = [
        json.loads(line)
        for line in (a.output / "benchmark-rank0.jsonl").read_text().splitlines()
    ]
    report = {
        "status": "pass",
        "hypothesis": spec["hypothesis"],
        "summary": summarize(rows, reference="relay_base"),
        "scope": "Eight exposed development questions,128-token cap; exploratory screen, no full-answer quality or confirmatory significance claim.",
    }
    (a.output / "research-result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
