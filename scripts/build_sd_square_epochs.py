"""Freeze the development-selected SD-square rate before a small-pool epoch screen."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.raw_root / "reports/mapper-scaling-20260905"
    registry_path = Path("reports/external-baselines-20260906/sd-square-decoding.json")
    result = json.loads(registry_path.read_text())
    fits = [base / "sd-square-fullpilot/run-27888", base / "sd-square-rates/run-27890"]
    with tempfile.TemporaryDirectory() as temporary:
        expected = Path(temporary) / "expected.json"
        subprocess.run(
            [
                sys.executable,
                "scripts/audit_sd_square_campaign.py",
                "--fits",
                *map(str, fits),
                "--runs",
                str(base / "sd-square-shards/run-27900"),
                str(base / "sd-square-shards/run-27901"),
                "--ledger",
                "reports/mapper-scaling-20260905/sd-square-shards/jobs.json",
                "--output",
                str(expected),
            ],
            check=True,
            capture_output=True,
        )
        if json.loads(expected.read_text()) != result:
            raise ValueError("SD-square development registry does not reproduce")
    name = max(
        (n for n, v in result["variants"].items() if v["kind"] == "steering"),
        key=lambda n: result["comparisons"]["native_ar"]["methods"][n][
            "tokens_per_second"
        ],
    )
    variant = result["variants"][name]
    gates = [
        json.loads((p / "sd-square-full-pilot-gate.json").read_text()) for p in fits
    ]
    selected = [
        fit
        for g in gates
        for fit in g["fitting"]
        if fit["checkpoint_sha256"] == variant["checkpoint_sha256"]
    ]
    if len(selected) != 1:
        raise ValueError("SD-square selection lacks unique original fitting state")
    fit = selected[0]
    config = json.loads(
        Path("configs/submission/baselines/sd-square-full-pilot.json").read_text()
    )
    config.update(
        worker_objectives=[variant["objective"]] * 4,
        learning_rate=fit["learning_rate"],
        learning_rate_end=fit["learning_rate_end"],
        worker_updates=[128, 256, 512, 1024],
        convergence_screen={
            "registry": str(registry_path),
            "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            "selected_method": name,
            "reproduction_trainable_sha256": variant["trainable_sha256"],
            "selection_rule": "highest throughput point among all six predeclared objective/rate settings",
        },
        scope="SD-square epoch-budget screen at fixed512 distinct examples. Four fresh fits use1,2,4,8 epochs, "
        "batch4 and128,256,512,1024updates. The objective/rate is frozen from the complete sixteen-request "
        "development screen before further fitting. Each horizon has its own public learning-rate decay "
        "schedule. These are separate fits, not checkpoints from one continued trajectory. Rank0 must "
        "reproduce the selected128-update training fingerprint exactly. No new data, quality or untouched "
        "confirmation claim. The10-minute pilot ceiling remains. Common decoding must compare all epoch endpoints.",
    )
    args.output.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
