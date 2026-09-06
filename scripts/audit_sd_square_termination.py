"""Rebuild the source-bound termination diagnosis from its raw worker traces."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    ledger_path = Path(
        "reports/mapper-scaling-20260905/sd-square-termination/jobs.json"
    )
    ledger = json.loads(ledger_path.read_text())
    config_path = Path(
        "configs/submission/baselines/sd-square-termination-diagnostic.json"
    )
    if (run / "source-commit.txt").read_text().strip() != ledger["source_commit"] or (
        run / "campaign-config.json"
    ).read_bytes() != config_path.read_bytes():
        raise ValueError("termination source/config differs from declared run")
    if [Path(j["local"]).name for j in ledger["jobs"]] != [run.name]:
        raise ValueError("termination job identity differs")
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        for path in run.glob("termination-rank*.json"):
            (folder / path.name).symlink_to(path)
        subprocess.run(
            [
                sys.executable,
                "scripts/check_sd_square_termination.py",
                "--config",
                str(config_path),
            ],
            env={**os.environ, "RELAYSPEC_OUTPUT": str(folder)},
            check=True,
            capture_output=True,
        )
        gate = json.loads((folder / "sd-square-numerical-gate.json").read_text())
        if gate != json.loads((run / "sd-square-numerical-gate.json").read_text()):
            raise ValueError("termination gate does not reproduce")
    paths = [
        ledger_path,
        config_path,
        run / "source-commit.txt",
        run / "campaign-config.json",
        run / "sd-square-numerical-gate.json",
        *sorted(run.glob("termination-rank*.json")),
    ]
    args.output.write_text(
        json.dumps(
            {
                "status": "complete",
                "input_sha256": {str(p): digest(p) for p in paths},
                "probes": gate["probes"],
                "scope": gate["scope"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
