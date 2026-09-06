"""Reproduce the prospective PARD gate and runtime-local throughput comparison."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from relayspec.ar_paper_evidence import read_rows, summarize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/submission/baselines/pard-campaign.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    ledger = json.loads(args.ledger.read_text())
    if (
        (run / "source-commit.txt").read_text().strip() != ledger["source_commit"]
        or len([j for j in ledger["jobs"] if Path(j["local"]).name == run.name]) != 1
        or args.config.read_bytes() != (run / "campaign-config.json").read_bytes()
    ):
        raise ValueError("PARD campaign source/job/config differs from declaration")
    config = json.loads(args.config.read_text())
    for path, sha in config.get("prerequisites", {}).items():
        prerequisite = Path(path)
        if (
            digest(prerequisite) != sha
            or json.loads(prerequisite.read_text())["status"] != "complete"
        ):
            raise ValueError("PARD longer-output prerequisite changed")
    protocol_path = Path(config["verification_protocol"])
    protocol = json.loads(protocol_path.read_text())
    source_config = Path(config["source_config"])
    setup = json.loads((run / "setup-gate.json").read_text())
    if (
        digest(protocol_path) != config["verification_protocol_sha256"]
        or any(digest(Path(p)) != sha for p, sha in protocol["input_sha256"].items())
        or digest(source_config) != config["source_config_sha256"]
        or setup["status"] != "pass"
        or setup["config_sha256"] != digest(source_config)
    ):
        raise ValueError("PARD prospective evidence or setup changed")
    raw_files = [*run.glob("benchmark-rank*.jsonl"), *run.glob("campaign-rank*.json")]
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for path in raw_files:
            (directory / path.name).symlink_to(path)
        env = os.environ.copy()
        env["RELAYSPEC_OUTPUT"] = str(directory)
        subprocess.run(
            [
                sys.executable,
                "scripts/check_pard_campaign.py",
                "--config",
                str(args.config),
            ],
            env=env,
            check=True,
            capture_output=True,
        )
        gate = json.loads((directory / "completion-gate.json").read_text())
        if gate != json.loads((run / "completion-gate.json").read_text()):
            raise ValueError("PARD completion gate does not reproduce")
    rows = read_rows(run)
    result = {
        "status": "complete",
        "input_sha256": {
            str(p): digest(p)
            for p in [
                args.config,
                args.ledger,
                protocol_path,
                source_config,
                run / "source-commit.txt",
                run / "setup-gate.json",
                run / "campaign-config.json",
                run / "completion-gate.json",
                *raw_files,
            ]
        },
        "comparison": summarize(rows),
        "ar_exact_count": gate["ar_exact_count"],
        "verification_replica_seconds": gate["verification_replica_seconds"],
        "original_exact_ar_gates": protocol["original_exact_ar_gates"],
        "scope": config["scope"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["comparison"]))


if __name__ == "__main__":
    main()
