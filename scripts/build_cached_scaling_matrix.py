"""Freeze the exact primary and regularization cell lists before running them."""

import argparse
import json
from pathlib import Path

from relayspec.scaling_matrix import (
    four_gpu_batches,
    primary_matrix,
    regularization_matrix,
    trial,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    groups = {
        "primary": primary_matrix()
        + [trial("dense", None, 128, seed=1730, study="seed_confirmation")],
        "regularization": regularization_matrix()
        + [
            trial("dense", None, 2048, seed=1730, study="seed_confirmation"),
            trial("dense", None, 32768, seed=1730, study="seed_confirmation"),
        ],
    }
    for group, cells in groups.items():
        for i, batch in enumerate(four_gpu_batches(cells)):
            (args.output / f"{group}-{i:02d}.json").write_text(
                json.dumps({"trials": batch}, indent=2) + "\n"
            )
    pilot = []
    for family, width in [
        ("factorized", 64),
        ("mlp", 64),
        ("factorized", 4096),
        ("mlp", 4096),
    ]:
        t = trial(family, width, 32768, study="capacity_resource_pilot")
        t.update(steps=16, checkpoint_steps=[16], budget_panels={})
        pilot.append(t)
    (args.output / "pilot.json").write_text(
        json.dumps({"trials": pilot}, indent=2) + "\n"
    )
    (args.output / "matrix.json").write_text(
        json.dumps(
            {
                "status": "predeclared",
                "groups": groups,
                "scope": "DFlash 4B to Qwen3-8B, Numina data, seed1729 primary. This is the cell declaration, not evidence of completed experiments.",
                "trajectory_reuse": "Both budget panels read their exact checkpoint from the same uninterrupted fit. N=8192 gives identical panels; no extra independent replicate is claimed.",
                "diagnostics": "1024 fixed validation records; first min(N,256) training records. Early checkpoints can include diagnostic training-pool records not yet seen by the optimizer.",
                "remaining": "Additional seeds, model/family replication, data composition, decoding/quality and adaptive research remain separate required studies.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
