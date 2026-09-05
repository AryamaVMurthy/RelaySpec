"""Declare the smaller-data study after the user paused large-data expansion."""

import argparse
import json
from pathlib import Path

from relayspec.scaling_matrix import four_gpu_batches, trial


def build():
    primary = [
        trial(family, width, n)
        for n in [512, 2048]
        for family, width in [("dense", None)]
        + [
            (f, w)
            for f in ["factorized", "mlp"]
            for w in [64, 128, 256, 512, 1024, 2048, 4096]
        ]
    ]
    reused = [t for t in primary if t.get("width") == 512]
    new = [t for t in primary if t not in reused]
    new += [
        trial("dense", None, 512, seed=s, study="seed_confirmation")
        for s in [1730, 1731]
    ]
    regularization = [
        trial(family, width, n, l2=l2, decay=decay, study="regularization")
        for n in [512, 2048]
        for family, width in [("dense", None), ("factorized", 512), ("mlp", 512)]
        for l2, decay in [(v, 0) for v in [1e-7, 1e-6, 1e-5, 1e-4]]
        + [(0, v) for v in [1e-4, 1e-3, 1e-2]]
    ]
    regularization += [
        trial("dense", None, 2048, seed=s, study="seed_confirmation")
        for s in [1730, 1731]
    ]
    for t in new + regularization:
        t.update(cache_backend="mmap", device_cache=True)
    # Obtain same-source dense controls early; subsequent batches group fit cost.
    dense = [t for t in new if t["architecture"] == "dense"]
    other = [t for t in new if t["architecture"] != "dense"]
    groups = {
        "primary": [dense] + four_gpu_batches(other),
        "regularization": four_gpu_batches(regularization),
    }
    return {
        "status": "predeclared",
        "distinct_examples": [512, 2048],
        "paused": "Further large-data expansion, including 32768, per user request.",
        "primary_cells": primary,
        "reuse": [
            {"trial": t, "job_id": 27726 if t["architecture"] == "mlp" else 27727}
            for t in reused
        ],
        "groups": groups,
        "scope": "30 primary capacity cells, four reused completed fits, plus four dense seed confirmations and 42 nonzero penalty cells. All new paths require pilot gates. Validation selection remains development; decoding and full-answer quality are separate.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = build()
    args.output.mkdir(parents=True, exist_ok=False)
    for group, batches in matrix["groups"].items():
        for i, batch in enumerate(batches):
            (args.output / f"{group}-{i:02d}.json").write_text(
                json.dumps({"trials": batch}, indent=2) + "\n"
            )
    (args.output / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")


if __name__ == "__main__":
    main()
