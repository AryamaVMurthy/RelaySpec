"""Run one shared-reference mapper campaign and verify duplicate-map isolation."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--equal-methods", nargs=2)
    args = parser.parse_args()
    python = os.environ["RELAYSPEC_PYTHON"]
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output / "config.yaml")
    subprocess.run(
        [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            "scripts/benchmark_relay.py",
            "--config",
            str(args.config),
        ],
        check=True,
    )
    subprocess.run(
        [
            python,
            "scripts/check_run_complete.py",
            "--config",
            str(args.config),
            "--output",
            str(output),
        ],
        check=True,
    )
    equality = None
    if args.equal_methods:
        rows = [
            json.loads(line)
            for p in output.glob("benchmark-rank*.jsonl")
            for line in p.read_text().splitlines()
        ]
        maps = [
            {(r["problem_id"], r["repetition"]): r for r in rows if r["method"] == name}
            for name in args.equal_methods
        ]
        if not maps[0] or maps[0].keys() != maps[1].keys():
            raise ValueError("duplicate mapper methods lack the same requests")
        for key, first in maps[0].items():
            second = maps[1][key]
            for field in [
                "output_hash",
                "output_tokens",
                "accepted_draft_lengths",
                "proposal_lengths",
                "mapper_checkpoint_sha256",
            ]:
                if first[field] != second[field]:
                    raise ValueError(
                        f"duplicate-map state isolation failed: {key} {field}"
                    )
        equality = {
            "methods": args.equal_methods,
            "requests": len(maps[0]),
            "status": "pass",
        }
    config = yaml.safe_load(args.config.read_text())
    (output / "campaign-gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "duplicate_map_equivalence": equality,
                "methods": config["benchmark"]["methods"],
                "scope": "Paired artifact completeness and optional identical-checkpoint isolation; not a task-quality or optimality claim.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
