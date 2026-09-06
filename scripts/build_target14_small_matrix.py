"""Freeze small-data 14B capacities without loading models or expanding data."""

import argparse
import hashlib
import json
from pathlib import Path

from relayspec.scaling_matrix import four_gpu_batches, trial, validate_trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("dflash", "eagle3"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = Path("configs/submission/scaling/target14b-small-v1/protocol.json")
    protocol = json.loads(protocol_path.read_text())
    declaration = protocol["proper_fits"]
    if (
        protocol["large_data_scaling"] != "paused"
        or declaration["distinct_examples"] != [512, 2048]
        or declaration["updates"] != 8192
        or declaration["batch_size"] != 4
    ):
        raise ValueError("small-data replication declaration changed")
    for name, sha in protocol["input_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
            raise ValueError("replication source configuration changed")
    primary = []
    for n in declaration["distinct_examples"]:
        for architecture in declaration["architectures"]:
            primary.append(trial(architecture["kind"], architecture.get("width"), n))
    # These seed controls measure reference variability, not duplicate padding.
    seeds = [trial("dense", None, n, seed=1730) for n in [512, 2048]]
    for cell in primary + seeds:
        cell.update(
            name=f"{args.family}14-{cell['name']}",
            study="target14_small_capacity"
            if cell["seed"] == 1729
            else "target14_dense_seed_control",
            normalize_input=declaration["normalization"][args.family],
            cache_backend="buffer",
            device_cache=False,
        )
        validate_trial(cell)
    pilot = []
    for kind, width in [
        ("factorized", 512),
        ("mlp", 512),
        ("factorized", 4096),
        ("mlp", 4096),
    ]:
        cell = trial(kind, width, 2048, study="target14_capacity_resource_pilot")
        cell.update(
            name=f"{args.family}14-pilot-{kind}{width}",
            steps=16,
            checkpoint_steps=[16],
            budget_panels={},
            normalize_input=declaration["normalization"][args.family],
            cache_backend="buffer",
            device_cache=False,
        )
        validate_trial(cell)
        pilot.append(cell)
    args.output.mkdir(parents=True, exist_ok=False)
    for index, batch in enumerate(four_gpu_batches(primary + seeds)):
        (args.output / f"batch-{index:02d}.json").write_text(
            json.dumps({"trials": batch}, indent=2) + "\n"
        )
    (args.output / "pilot.json").write_text(
        json.dumps({"trials": pilot}, indent=2) + "\n"
    )
    (args.output / "matrix.json").write_text(
        json.dumps(
            {
                "status": "declared_before_fitting",
                "family": args.family,
                "target": protocol["target"],
                "large_data_scaling": "paused",
                "protocol_sha256": hashlib.sha256(
                    protocol_path.read_bytes()
                ).hexdigest(),
                "primary_cells": primary,
                "dense_seed_controls": seeds,
                "cache_binding": "Execution requires a passed capacity pilot on the exact cache-index SHA. No cache is inferred or substituted.",
                "batching": "Existing four_gpu_batches orders cost. Its8B parameter proxy only orders jobs, never changes fitting or reported14B parameter counts.",
                "scope": "14 primary cells and two distinct-seed dense controls per family. Fixed32,768 presentations mean repeated512/2048-example pools, not32,768 distinct examples. The secondary four-pass panel uses its exact saved checkpoint. Full decoding, quality and selected three-seed replication remain required.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
