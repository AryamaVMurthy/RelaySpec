"""Gate a four-candidate cached fit; optionally run a bounded decoding pilot."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch
import yaml

from relayspec.scaling_matrix import validate_cache_access_gate, validate_trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--mode", choices=["pilot", "fit"], required=True)
    parser.add_argument("--pilot-gate", type=Path, required=True)
    parser.add_argument("--cache-access-gate", type=Path)
    parser.add_argument("--continuation-gate", type=Path)
    args = parser.parse_args()
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    cache = Path(os.environ["RELAYSPEC_FEATURE_CACHE"])
    index_hash = hashlib.sha256((cache / "cache-index.json").read_bytes()).hexdigest()
    pilot = json.loads(args.pilot_gate.read_text())
    if pilot["status"] != "pass":
        raise ValueError("a successful pilot gate is required")
    if args.mode == "fit" and pilot.get("feature_cache_index_sha256") != index_hash:
        raise ValueError("full fits require the capacity pilot on this exact cache")
    trials = json.loads(args.trials.read_text())["trials"]
    access_gate = (
        json.loads(args.cache_access_gate.read_text())
        if args.cache_access_gate
        else None
    )
    validate_cache_access_gate(trials, access_gate, index_hash)
    if any(t.get("resume_from") for t in trials):
        if not args.continuation_gate:
            raise ValueError("continued fitting requires the exact continuation pilot")
        continued = json.loads(args.continuation_gate.read_text())
        if (
            continued.get("status") != "pass"
            or continued.get("feature_cache_index_sha256") != index_hash
            or len(continued.get("pairs", [])) != 4
            or any(p.get("bit_identical") is not True for p in continued["pairs"])
        ):
            raise ValueError("continuation pilot did not pass on this cache")
    if len(trials) != 4 or len({t["name"] for t in trials}) != 4:
        raise ValueError("batch requires four unique candidate trials")
    for trial in trials:
        validate_trial(trial)
        if args.mode == "pilot" and trial["steps"] > 16:
            raise ValueError("capacity resource pilot is capped at 16 updates")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.trials, output / "trials.json")
    fits = Path(os.environ["RELAYSPEC_FIT_OUTPUT"])
    python = os.environ["RELAYSPEC_PYTHON"]
    command = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        "scripts/fit_cached_mappers.py",
        "--trials",
        str(args.trials),
    ]
    if args.mode == "pilot":
        command.append("--equivalence-pilot")
    started = time.perf_counter()
    subprocess.run(
        command, env={**os.environ, "RELAYSPEC_OUTPUT": str(fits)}, check=True
    )
    fit_wall = time.perf_counter() - started
    checkpoint_hashes = {}
    for trial in trials:
        folder = fits / trial["name"]
        complete = json.loads((folder / "fit-complete.json").read_text())
        if (
            complete["status"] != "pass"
            or complete["trial"] != trial
            or complete["feature_cache_index_sha256"] != index_hash
        ):
            raise ValueError("fitting result does not match the declared trial/cache")
        if complete["distinct_records_seen"] != min(
            trial["distinct_examples"], trial["steps"] * 4
        ):
            raise ValueError("fitting result miscounts distinct data")
        continuation = folder / "continuation.pt"
        if (
            hashlib.sha256(continuation.read_bytes()).hexdigest()
            != complete["continuation_sha256"]
        ):
            raise ValueError("continuation state hash mismatch")
        resume = torch.load(continuation, weights_only=True, map_location="cpu")
        if (
            resume["completed_trial"] != trial
            or resume["steps"] != trial["steps"]
            or not resume["optimizer"]["state"]
        ):
            raise ValueError("continuation did not preserve the fitted optimizer")
        if any(
            not torch.isfinite(t).all()
            for state in resume["optimizer"]["state"].values()
            for t in state.values()
            if isinstance(t, torch.Tensor)
        ):
            raise ValueError("nonfinite continuation optimizer state")
        del resume
        for step in trial["checkpoint_steps"]:
            path = folder / f"step-{step:06d}.pt"
            state = torch.load(path, weights_only=True, map_location="cpu")
            if state["steps"] != step or any(
                not torch.isfinite(v).all() for v in state["relay"].values()
            ):
                raise ValueError("checkpoint step or finite-weight gate failed")
            checkpoint_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            del state
        destination = output / "fitting" / trial["name"]
        destination.mkdir(parents=True, exist_ok=False)
        for path in folder.iterdir():
            if path.suffix != ".pt":
                shutil.copy2(path, destination / path.name)
    if args.mode == "pilot":
        config = yaml.safe_load(
            Path("configs/submission/scaling/campaign-pilot.yaml").read_text()
        )
        variants = {
            f"relay_{i}": str(fits / t["name"] / "step-000016.pt")
            for i, t in enumerate(trials)
        }
        variants["relay_duplicate"] = variants["relay_0"]
        config["relay_probe"]["variants"] = variants
        config["benchmark"]["methods"] = [
            "native_ar",
            "native_target_dflash",
            "optimized_source_reuse",
            *variants,
        ]
        path = output / "campaign-config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        subprocess.run(
            [
                python,
                "scripts/run_mapper_campaign.py",
                "--config",
                str(path),
                "--equal-methods",
                "relay_0",
                "relay_duplicate",
            ],
            env={**os.environ, "RELAYSPEC_OUTPUT": str(output / "evaluation")},
            check=True,
        )
    (output / "batch-gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "mode": args.mode,
                "trials": [t["name"] for t in trials],
                "trials_sha256": hashlib.sha256(args.trials.read_bytes()).hexdigest(),
                "feature_cache_index_sha256": index_hash,
                "pilot_gate_sha256": hashlib.sha256(
                    args.pilot_gate.read_bytes()
                ).hexdigest(),
                "cache_access_gate_sha256": (
                    hashlib.sha256(args.cache_access_gate.read_bytes()).hexdigest()
                    if args.cache_access_gate
                    else None
                ),
                "continuation_gate_sha256": (
                    hashlib.sha256(args.continuation_gate.read_bytes()).hexdigest()
                    if args.continuation_gate
                    else None
                ),
                "checkpoint_sha256": checkpoint_hashes,
                "fit_wall_seconds": fit_wall,
                "total_wall_seconds": time.perf_counter() - started,
                "scope": "Declared cached fitting/checkpoint completion. Pilot additionally checks BF16 batch equivalence, streaming, width extremes and duplicate-map decoding. This is not full task-quality evidence.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
