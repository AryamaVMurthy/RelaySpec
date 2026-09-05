"""Compare unchanged fitting against memory-mapped and GPU-resident inputs."""

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch


def assert_identical(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
            raise ValueError(f"cache optimization changed {path}")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError(f"cache optimization changed keys at {path}")
        for key in left:
            assert_identical(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise ValueError(f"cache optimization changed length at {path}")
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            assert_identical(a, b, f"{path}.{i}")
    elif left != right:
        raise ValueError(f"cache optimization changed {path}: {left} != {right}")


def main():
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    fits = Path(os.environ["RELAYSPEC_FIT_OUTPUT"])
    cache = Path(os.environ["RELAYSPEC_FEATURE_CACHE"])
    gate = json.loads(Path(os.environ["RELAYSPEC_PILOT_GATE"]).read_text())
    cache_hash = hashlib.sha256((cache / "cache-index.json").read_bytes()).hexdigest()
    if gate["status"] != "pass" or gate["feature_cache_index_sha256"] != cache_hash:
        raise ValueError("cache access pilot requires the completed capacity gate")
    trials = []
    for name, architecture, n, steps, backend, device_cache, l2 in [
        ("stream-buffer", "mlp", 4096, 2048, "buffer", False, 1e-6),
        ("stream-mmap", "mlp", 4096, 2048, "mmap", False, 1e-6),
        ("dense-host", "dense", 512, 1024, "buffer", False, 0),
        ("dense-device", "dense", 512, 1024, "mmap", True, 0),
    ]:
        trial = {
            "name": name,
            "architecture": architecture,
            "distinct_examples": n,
            "steps": steps,
            "seed": 1729,
            "learning_rate": 6e-4,
            "l2_weight": l2,
            "weight_decay": 0.0,
            "validation_records": 32,
            "diagnostic_train_records": 32,
            "checkpoint_steps": [1, steps],
            "cache_backend": backend,
            "device_cache": device_cache,
        }
        if architecture == "mlp":
            trial["width"] = 512
        trials.append(trial)
    output.mkdir(parents=True, exist_ok=True)
    config = output / "trials.json"
    config.write_text(json.dumps({"trials": trials}, indent=2) + "\n")
    started = time.perf_counter()
    subprocess.run(
        [
            os.environ["RELAYSPEC_PYTHON"],
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            "scripts/fit_cached_mappers.py",
            "--trials",
            str(config),
        ],
        env={**os.environ, "RELAYSPEC_OUTPUT": str(fits)},
        check=True,
    )
    timings = {}
    for trial in trials:
        folder = fits / trial["name"]
        complete = json.loads((folder / "fit-complete.json").read_text())
        if complete["status"] != "pass" or complete["trial"] != trial:
            raise ValueError("cache access trial did not complete its budget")
        timings[trial["name"]] = {
            k: complete[k]
            for k in [
                "loop_seconds",
                "setup_seconds",
                "input_io_seconds",
                "validation_seconds",
                "peak_gpu_bytes",
            ]
        }
        destination = output / "fitting" / trial["name"]
        destination.mkdir(parents=True, exist_ok=False)
        for path in folder.iterdir():
            if path.suffix != ".pt":
                shutil.copy2(path, destination / path.name)
    pairs = []
    for first, second in [
        ("stream-buffer", "stream-mmap"),
        ("dense-host", "dense-device"),
    ]:
        a, b = [
            torch.load(
                fits / name / "continuation.pt", weights_only=True, map_location="cpu"
            )
            for name in [first, second]
        ]
        for key in [
            "relay",
            "optimizer",
            "steps",
            "tokens_seen",
            "torch_rng_state",
            "cuda_rng_state",
        ]:
            assert_identical(a[key], b[key], key)
        del a, b
        for name in ["training.jsonl", "validation.jsonl"]:
            if (fits / first / name).read_bytes() != (
                fits / second / name
            ).read_bytes():
                raise ValueError(f"cache optimization changed {name}: {first}/{second}")
        pairs.append(
            {
                "baseline": first,
                "candidate": second,
                "weights_optimizer_and_loss_trajectory_bit_identical": True,
                "loop_speedup": timings[first]["loop_seconds"]
                / timings[second]["loop_seconds"],
            }
        )
    (output / "cache-access-gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "feature_cache_index_sha256": cache_hash,
                "pairs": pairs,
                "timings": timings,
                "wall_seconds": time.perf_counter() - started,
                "scope": "Unchanged data/order/masking/optimizer and bit-identical weights, optimizer states and every recorded loss/validation point. Cache access timing only; no new decoding performance claim.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
