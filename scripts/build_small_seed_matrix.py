"""Declare seed replications for selected small-data deployment/convergence points."""

import argparse
import json
from pathlib import Path

from relayspec.scaling_matrix import four_gpu_batches, trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells = []
    for seed in (1730, 1731):
        for architecture, width, n, rate in (
            ("factorized", 1024, 512, 6e-4),
            ("factorized", 4096, 2048, 2e-4),
            ("mlp", 4096, 2048, 2e-4),
            ("mlp", 4096, 512, 2e-4),
        ):
            cell = trial(
                architecture,
                width,
                n,
                seed=seed,
                study="selected_small_seed_replication",
            )
            cell.update(cache_backend="mmap", device_cache=True, learning_rate=rate)
            if rate != 6e-4:
                cell["name"] += f"-lr-{rate:g}"
            cells.append(cell)
    batches = four_gpu_batches(cells)
    args.output.mkdir(parents=True, exist_ok=False)
    for i, batch in enumerate(batches):
        (args.output / f"batch-{i:02d}.json").write_text(
            json.dumps({"trials": batch}, indent=2) + "\n"
        )
    (args.output / "matrix.json").write_text(
        json.dumps(
            {
                "status": "predeclared",
                "distinct_examples": [512, 2048],
                "groups": {"selected_seeds": batches},
                "reuse": [],
                "reference_seed": 1729,
                "new_seeds": [1730, 1731],
                "selection_evidence": [
                    "reports/mapper-scaling-20260905/capacity-decoding-results.json",
                    "reports/mapper-scaling-20260905/learning-rate-results.json",
                ],
                "scope": "Eight replications at two additional seeds. Factor1024/N512 is a "
                "compact deployment candidate from exposed development decoding. "
                "Wide4096 factor/MLP atN2048 use the lowest feature-validation learning "
                "rate among the three declared rates. WideMLP/N512 tests repeatability "
                "of the observed early validation minimum. Save all declared checkpoints. "
                "Report fixed8192 endpoints and per-seed validation-selected checkpoints "
                "separately. No fitting-data expansion, no untouched confirmation use, "
                "no claim of optimality from the reference seed alone.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Declared {len(cells)} new seed fits in {len(batches)} four-GPU batches")


if __name__ == "__main__":
    main()
