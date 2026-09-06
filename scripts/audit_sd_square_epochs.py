"""Reproduce the selected-rate epoch screen and its fixed-pool fitting costs."""

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
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    config_path = Path("configs/submission/baselines/sd-square-epochs.json")
    ledger_path = Path("reports/mapper-scaling-20260905/sd-square-epochs/jobs.json")
    config = json.loads(config_path.read_text())
    ledger = json.loads(ledger_path.read_text())
    if (run / "source-commit.txt").read_text().strip() != ledger[
        "source_commit"
    ] or len([j for j in ledger["jobs"] if Path(j["local"]).name == run.name]) != 1:
        raise ValueError("SD-square epoch source/job differs from launch record")
    raw = [
        run / f"{kind}-rank{rank}.json"
        for rank in range(4)
        for kind in ("pilot", "training", "decoding")
    ]
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        expected = folder / "config.json"
        subprocess.run(
            [
                sys.executable,
                "scripts/build_sd_square_epochs.py",
                "--raw-root",
                str(args.raw_root),
                "--output",
                str(expected),
            ],
            check=True,
            capture_output=True,
        )
        if expected.read_bytes() != config_path.read_bytes():
            raise ValueError("SD-square epoch declaration does not reproduce")
        for path in raw:
            (folder / path.name).symlink_to(path)
        env = os.environ.copy()
        env["RELAYSPEC_OUTPUT"] = str(folder)
        subprocess.run(
            [
                sys.executable,
                "scripts/summarize_sd_square_fullpilot.py",
                "--config",
                str(config_path),
            ],
            env=env,
            check=True,
            capture_output=True,
        )
        gate_path = run / "sd-square-full-pilot-gate.json"
        gate = json.loads(gate_path.read_text())
        if json.loads((folder / gate_path.name).read_text()) != gate:
            raise ValueError("SD-square epoch completion gate does not reproduce")
    records = []
    for fit in gate["fitting"]:
        history = fit["training"]
        records.append(
            {
                k: fit[k]
                for k in (
                    "rank",
                    "objective",
                    "seed",
                    "updates",
                    "epochs",
                    "learning_rate",
                    "learning_rate_end",
                    "training_seconds",
                    "distinct_records_seen",
                    "trainable_parameters",
                    "trainable_sha256",
                    "checkpoint",
                    "checkpoint_sha256",
                )
            }
        )
        records[-1].update(
            supervised_positions=sum(s["loss_tokens"] for s in history),
            epoch_mean_training_losses=[
                sum(s["loss"] for s in history[start : start + 128]) / 128
                for start in range(0, fit["updates"], 128)
            ],
        )
    inputs = [
        config_path,
        ledger_path,
        Path(config["convergence_screen"]["registry"]),
        run / "source-commit.txt",
        gate_path,
        *raw,
    ]
    result = {
        "status": "complete",
        "input_sha256": {str(p): digest(p) for p in inputs},
        "selection": config["convergence_screen"],
        "records": records,
        "original_exact_ar_gates": gate["original_exact_ar_gates"],
        "scope": config["scope"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(records))


if __name__ == "__main__":
    main()
