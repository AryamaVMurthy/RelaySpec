"""Reproduce measured initialization and dense-reference adaptation budgets."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    declaration = Path("configs/submission/scaling/baseline-time-calibration.json")
    trials = json.loads(declaration.read_text())["trials"]
    gate_path = args.run / "batch-gate.json"
    gate = json.loads(gate_path.read_text())
    if (
        gate["status"] != "pass"
        or gate["trials_sha256"] != digest(declaration)
        or len(gate["reference_checkpoint_checks"]) != 4
    ):
        raise ValueError("timing calibration is missing exact-reference checks")
    inputs = {str(declaration): digest(declaration), str(gate_path): digest(gate_path)}
    rows = []
    fields = [
        "setup_seconds",
        "initial_validation_seconds",
        "training_update_seconds",
        "validation_seconds",
        "checkpoint_export_seconds",
        "loop_seconds",
        "worker_total_seconds",
    ]
    for trial, check in zip(trials, gate["reference_checkpoint_checks"], strict=True):
        path = args.run / "fitting" / trial["name"] / "fit-complete.json"
        fit = json.loads(path.read_text())
        if (
            fit["status"] != "pass"
            or fit["trial"] != trial
            or fit["distinct_records_seen"] != 512
            or check
            != {
                "trial": trial["name"],
                "step": trial["steps"],
                "reference_sha256": trial["reference_checkpoint"]["sha256"],
                "weights_bit_identical": True,
            }
            or any(not math.isfinite(fit[k]) or fit[k] <= 0 for k in fields)
            or fit["training_update_seconds"]
            + fit["validation_seconds"]
            + fit["checkpoint_export_seconds"]
            > fit["loop_seconds"] + 0.01
            or fit["setup_seconds"]
            + fit["initial_validation_seconds"]
            + fit["loop_seconds"]
            > fit["worker_total_seconds"]
        ):
            raise ValueError("timing components or reference identity are inconsistent")
        inputs[str(path)] = digest(path)
        rows.append(
            {
                "trial": trial["name"],
                "updates": trial["steps"],
                **{k: fit[k] for k in fields},
            }
        )
    summaries = {}
    for steps in (128, 8192):
        group = [r for r in rows if r["updates"] == steps]
        if len(group) != 2:
            raise ValueError("two measured repetitions are required at each endpoint")
        summaries[str(steps)] = {
            k: {
                "mean": statistics.mean(r[k] for r in group),
                "range": [min(r[k] for r in group), max(r[k] for r in group)],
            }
            for k in fields
        }
    baseline = summaries["8192"]["training_update_seconds"]["mean"]
    initial = summaries["128"]["training_update_seconds"]["mean"]
    result = {
        "status": "complete",
        "input_sha256": inputs,
        "measurements": rows,
        "summaries": summaries,
        "warm_training_budgets": [
            {
                "multiplier": multiplier,
                "total_training_seconds": multiplier * baseline,
                "initial_mapper_charge_seconds": initial,
                "remaining_update_seconds": multiplier * baseline - initial,
            }
            for multiplier in (0.25, 1.0, 4.0)
        ],
        "scope": "Two exact-weight repetitions of each endpoint on the same 512 cached "
        "records. Warm training includes access, forward/backward, optimizer and log "
        "writes, excluding validation and checkpoint export. Initialization is charged "
        "to the warm training budget. Worker totals report setup/validation/export "
        "separately and still exclude feature-cache construction and Python import "
        "startup. Fresh-setup cost comparisons must add all required costs, including "
        "teacher-label preparation. The two-repetition ranges are not confidence intervals.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["warm_training_budgets"]))


if __name__ == "__main__":
    main()
