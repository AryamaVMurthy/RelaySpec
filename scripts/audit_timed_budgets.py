"""Audit first-crossing time budgets and preserve complete setup accounting."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = Path("configs/submission/scaling/adaptation-budget-protocol.json")
    protocol = json.loads(protocol_path.read_text())
    timing_path = Path(
        "reports/external-baselines-20260906/baseline-time-calibration.json"
    )
    timing = json.loads(timing_path.read_text())
    if protocol["input_sha256"][str(timing_path)] != digest(timing_path):
        raise ValueError("measured initialization timing differs from the protocol")
    initial_worker_seconds = timing["summaries"]["128"]["worker_total_seconds"]["mean"]
    inputs = {
        str(protocol_path): digest(protocol_path),
        str(timing_path): digest(timing_path),
    }
    records, multipliers, gates = [], set(), {}
    for run in args.runs:
        path = run / "timed-budget-gate.json"
        gate = json.loads(path.read_text())
        multiplier = gate["budget"]["multiplier"]
        budget = next(
            b
            for b in protocol["warm_training_budgets"]
            if b["multiplier"] == multiplier
        )
        if (
            multiplier in multipliers
            or gate["status"] != "pass"
            or gate["budget"] != budget
            or gate["protocol_sha256"] != digest(protocol_path)
        ):
            raise ValueError("time-budget run is duplicated or changes the protocol")
        multipliers.add(multiplier)
        gates[multiplier] = {"sha256": digest(path), "gate": gate}
        inputs[str(path)] = digest(path)
        for filename, key in [
            ("fitting-config.yaml", "config_sha256"),
            ("feature-trial.json", "feature_trial_sha256"),
        ]:
            if digest(run / filename) != gate[key]:
                raise ValueError("timed fitting declaration changed")
            inputs[str(run / filename)] = digest(run / filename)
        if len(gate["fitting"]) != 4:
            raise ValueError("timed fitting is missing workers")
        for rank, summary in enumerate(gate["fitting"]):
            folder = run / "fitting" / ("feature" if rank == 0 else f"rank{rank}")
            fit_path = folder / (
                "fit-complete.json" if rank == 0 else "adaptation-fit-gate.json"
            )
            fit = json.loads(fit_path.read_text())
            history_path = folder / (
                "budget-timing.jsonl" if rank == 0 else "training.jsonl"
            )
            history = [
                json.loads(line) for line in history_path.read_text().splitlines()
            ]
            values = [r["training_update_seconds"] for r in history]
            expected = (
                budget["total_training_seconds"]
                if rank == 0
                else budget["remaining_update_seconds"]
            )
            charge = 0 if rank == 0 else budget["initial_mapper_charge_seconds"]
            if (
                summary["rank"] != rank
                or summary["fit_gate"] != fit
                or summary["fit_gate_sha256"] != digest(fit_path)
                or summary["budget_seconds"] != expected
                or summary["initial_mapper_training_charge_seconds"] != charge
                or [r["step"] for r in history]
                != list(range(1, summary["updates"] + 1))
                or values[-1] != summary["training_update_seconds"]
                or (values[-2] if len(values) > 1 else 0)
                != summary["previous_update_seconds"]
                or any(a >= b for a, b in zip([0.0, *values[:-1]], values, strict=True))
                or not summary["previous_update_seconds"] < expected <= values[-1]
                or any(value >= expected for value in values[:-1])
                or summary["additional_records_seen"]
                != min(512, summary["updates"] * 4)
                or summary["additional_presentations"] != summary["updates"] * 4
            ):
                raise ValueError(
                    "raw records do not establish first-crossing budget stopping"
                )
            inputs.update(
                {
                    str(fit_path): digest(fit_path),
                    str(history_path): digest(history_path),
                }
            )
            own_worker = (
                fit["worker_total_seconds"] if rank == 0 else fit["total_seconds"]
            )
            records.append(
                {
                    "multiplier": multiplier,
                    "method": summary["method"],
                    "rank": rank,
                    "updates": summary["updates"],
                    "additional_records_seen": summary["additional_records_seen"],
                    "additional_presentations": summary["additional_presentations"],
                    "total_record_presentations_with_initial_mapper": summary[
                        "additional_presentations"
                    ]
                    + (0 if rank == 0 else 512),
                    "total_distinct_training_records": 512,
                    "warm_training_seconds_with_initial_mapper": values[-1] + charge,
                    "nominal_budget_seconds": budget["total_training_seconds"],
                    "overshoot_seconds": values[-1] - expected,
                    "worker_seconds_with_initial_mapper": own_worker
                    + (0 if rank == 0 else initial_worker_seconds),
                    "initial_mapper_worker_charge_seconds": 0
                    if rank == 0
                    else initial_worker_seconds,
                    "fit_gate": fit,
                    "checkpoint": summary["checkpoint"],
                    "checkpoint_sha256": summary["checkpoint_sha256"],
                }
            )
    if 0.25 in gates:
        if not gates[0.25]["gate"]["pilot_decoding_passed"]:
            raise ValueError("quarter-budget pilot did not decode successfully")
        for multiplier, item in gates.items():
            if (
                multiplier != 0.25
                and item["gate"]["pilot_gate_sha256"] != gates[0.25]["sha256"]
            ):
                raise ValueError("full fit uses another timed-checkpoint pilot")
    result = {
        "status": "complete" if multipliers == {0.25, 1.0, 4.0} else "partial",
        "completed_multipliers": sorted(multipliers),
        "input_sha256": inputs,
        "records": records,
        "scope": "Measured warm-training budget fits only. Per-update traces prove first-crossing "
        "stopping and at-most-one-update overshoot. Worker totals include the measured initial "
        "mapper worker cost for CE/LoRA, but exclude Python import startup and shared feature-cache "
        "construction/storage. Record full Slurm allocation and decoding separately. Only the "
        "quarter-budget correctness pilot includes decoding here. Full paired budget decoding "
        "and task quality are separate required evidence. No large-data expansion.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_multipliers": sorted(multipliers),
                "records": [
                    {
                        k: r[k]
                        for k in [
                            "multiplier",
                            "method",
                            "updates",
                            "warm_training_seconds_with_initial_mapper",
                            "overshoot_seconds",
                        ]
                    }
                    for r in records
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
