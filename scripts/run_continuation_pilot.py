"""Require split fitting to reproduce uninterrupted weights and every loss."""

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch
from run_cache_access_pilot import assert_identical

from relayspec.scaling_matrix import validate_cache_access_gate


def main():
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    fits = Path(os.environ["RELAYSPEC_FIT_OUTPUT"])
    cache = Path(os.environ["RELAYSPEC_FEATURE_CACHE"])
    cache_hash = hashlib.sha256((cache / "cache-index.json").read_bytes()).hexdigest()
    gate = json.loads(Path(os.environ["RELAYSPEC_PILOT_GATE"]).read_text())
    if gate["status"] != "pass" or gate["feature_cache_index_sha256"] != cache_hash:
        raise ValueError("continuation pilot needs this cache's capacity gate")
    access = json.loads(Path(os.environ["RELAYSPEC_CACHE_ACCESS_GATE"]).read_text())
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base = dict(
        distinct_examples=512,
        seed=1729,
        learning_rate=6e-4,
        l2_weight=1e-6,
        weight_decay=0.0,
        validation_records=32,
        diagnostic_train_records=32,
        cache_backend="mmap",
        device_cache=True,
    )
    first = []
    for family in ["dense", "mlp"]:
        for name, steps in [("full", 1024), ("prefix", 512)]:
            t = {
                **base,
                "name": f"{family}-{name}",
                "architecture": family,
                "steps": steps,
                "checkpoint_steps": [512, 1024] if steps == 1024 else [512],
            }
            if family == "mlp":
                t["width"] = 512
            first.append(t)

    def run(trials, name):
        validate_cache_access_gate(trials, access, cache_hash)
        config = output / f"{name}-trials.json"
        config.write_text(json.dumps({"trials": trials}, indent=2) + "\n")
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
        for trial in trials:
            folder = fits / trial["name"]
            complete = json.loads((folder / "fit-complete.json").read_text())
            if complete["status"] != "pass" or complete["trial"] != trial:
                raise ValueError("continuation pilot fit did not finish")
            destination = output / "fitting" / trial["name"]
            destination.mkdir(parents=True, exist_ok=False)
            for path in folder.iterdir():
                if path.suffix != ".pt":
                    shutil.copy2(path, destination / path.name)

    run(first, "first")
    second = []
    for trial in first:
        if trial["steps"] != 512:
            continue
        parent = fits / trial["name"] / "continuation.pt"
        for suffix in ["a", "b"]:
            second.append(
                {
                    **trial,
                    "name": f"{trial['architecture']}-resumed-{suffix}",
                    "steps": 1024,
                    "checkpoint_steps": [1024],
                    "resume_from": str(parent),
                    "resume_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
                }
            )
    run(second, "second")
    pairs = []
    for trial in second:
        family = trial["architecture"]
        full, prefix, resumed = [
            fits / n for n in [f"{family}-full", f"{family}-prefix", trial["name"]]
        ]
        a, b = [
            torch.load(p / "continuation.pt", weights_only=True, map_location="cpu")
            for p in [full, resumed]
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
        if (full / "training.jsonl").read_bytes() != (
            prefix / "training.jsonl"
        ).read_bytes() + (resumed / "training.jsonl").read_bytes():
            raise ValueError("continuation changed the training trajectory")
        original = {
            p["step"]: p
            for p in map(
                json.loads, (full / "validation.jsonl").read_text().splitlines()
            )
        }
        for part in [prefix, resumed]:
            for p in map(
                json.loads, (part / "validation.jsonl").read_text().splitlines()
            ):
                if p != original[p["step"]]:
                    raise ValueError("continuation changed validation")
        pairs.append(
            {"uninterrupted": full.name, "resumed": resumed.name, "bit_identical": True}
        )
    (output / "continuation-gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "feature_cache_index_sha256": cache_hash,
                "pairs": pairs,
                "wall_seconds": time.perf_counter() - started,
                "scope": "Exact interrupted/uninterrupted fitting equivalence, including explicit L2, Adam state, sample position, RNG and every loss. No extra distinct data or decoding claim.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
