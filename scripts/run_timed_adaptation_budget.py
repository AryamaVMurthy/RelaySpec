"""Fit the frozen measured-time protocol; the first quarter-budget run also decodes."""

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multiplier", type=float, choices=[0.25, 1.0, 4.0], required=True
    )
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-gate", type=Path)
    args = parser.parse_args()
    protocol_path = Path("configs/submission/scaling/adaptation-budget-protocol.json")
    protocol = json.loads(protocol_path.read_text())
    if args.pilot and args.multiplier != 0.25:
        raise ValueError("the first timed-checkpoint pilot must use the quarter budget")
    if not args.pilot:
        if args.pilot_gate is None:
            raise ValueError(
                "full timed fits require a completed timed-checkpoint pilot"
            )
        pilot = json.loads(args.pilot_gate.read_text())
        if (
            pilot["status"] != "pass"
            or not pilot["pilot_decoding_passed"]
            or pilot["protocol_sha256"] != digest(protocol_path)
        ):
            raise ValueError("timed-checkpoint pilot did not pass this protocol")
    budget = next(
        b
        for b in protocol["warm_training_budgets"]
        if b["multiplier"] == args.multiplier
    )
    prerequisite = Path(os.environ["ADAPTATION_CONTINUATION_GATE"])
    resume = json.loads(prerequisite.read_text())
    expected_resume = [
        sha
        for name, sha in protocol["input_sha256"].items()
        if name.endswith("/adaptation-continuation-gate.json")
    ]
    if expected_resume != [digest(prerequisite)]:
        raise ValueError("continuation gate differs from the frozen budget protocol")
    if (
        resume["status"] != "pass"
        or not resume["decoding_bit_identical"]
        or any(not p["weights_and_optimizer_bit_identical"] for p in resume["pairs"])
    ):
        raise ValueError("exact adaptation continuation has not passed")
    config = yaml.safe_load(
        Path(
            "configs/submission/scaling/drafter-adaptation-512-calibration.yaml"
        ).read_text()
    )
    settings = config["adaptation_pilot"]
    settings.update(
        pilot_updates=65536, diagnostics=False, check_direct_fusion_equivalence=False
    )
    settings["worker_trials"] = [
        {"name": "feature_dispatch", "lora_rank": 0, "learning_rate": 0.0006},
        {
            "name": "connector_ce",
            "lora_rank": 0,
            "learning_rate": protocol["selected_rates"]["ce"]["learning_rate"],
            "seed": 1729,
        },
        {
            "name": "lora32_s1729",
            "lora_rank": 32,
            "learning_rate": protocol["selected_rates"]["lora32"]["learning_rate"],
            "seed": 1729,
        },
        {
            "name": "lora32_s1730",
            "lora_rank": 32,
            "learning_rate": protocol["selected_rates"]["lora32"]["learning_rate"],
            "seed": 1730,
        },
    ]
    for worker in settings["worker_trials"][1:]:
        worker["warm_time_budget_seconds"] = budget["remaining_update_seconds"]
    feature = copy.deepcopy(
        json.loads(
            Path(
                "configs/submission/scaling/matrix-focused-v1/primary-00.json"
            ).read_text()
        )["trials"][0]
    )
    feature.update(
        name="feature",
        steps=65536,
        checkpoint_steps=[],
        budget_panels={},
        study="matched_time_budget",
        warm_time_budget_seconds=budget["total_training_seconds"],
    )
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    fits = Path(os.environ["RELAYSPEC_FIT_OUTPUT"])
    python = os.environ["RELAYSPEC_PYTHON"]
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "fitting-config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    trials_path = output / "feature-trial.json"
    trials_path.write_text(json.dumps({"trials": [feature]}, indent=2) + "\n")
    declaration = {
        "protocol_sha256": digest(protocol_path),
        "budget": budget,
        "config_sha256": digest(config_path),
        "feature_trial_sha256": digest(trials_path),
        "continuation_gate_sha256": digest(prerequisite),
        "pilot": args.pilot,
        "pilot_gate_sha256": digest(args.pilot_gate) if args.pilot_gate else None,
    }
    (output / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    started = time.perf_counter()
    subprocess.run(
        [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            "scripts/fit_timed_budget_worker.py",
            "--config",
            str(config_path),
            "--trials",
            str(trials_path),
        ],
        check=True,
    )
    fit_wall_seconds = time.perf_counter() - started
    summaries = []
    variants = {"relay_initial": settings["initial_mapper"]}
    drafter_updates = {}
    for rank in range(4):
        folder = fits / ("feature" if rank == 0 else f"rank{rank}")
        gate_name = "fit-complete.json" if rank == 0 else "adaptation-fit-gate.json"
        gate = json.loads((folder / gate_name).read_text())
        expected = (
            budget["total_training_seconds"]
            if rank == 0
            else budget["remaining_update_seconds"]
        )
        actual = gate["training_update_seconds"]
        previous = gate["previous_update_seconds"]
        if (
            gate["status"] != "pass"
            or gate["warm_time_budget_seconds"] != expected
            or not 0 <= previous < expected <= actual
        ):
            raise ValueError("fit did not stop at its first budget-crossing update")
        if rank == 0:
            if gate["trial"] != feature:
                raise ValueError("timed feature fit changed its declaration")
            updates = gate["steps"]
            path = folder / f"step-{updates:06d}.pt"
            name = "relay_feature"
            variants[name] = str(path)
            prepared, seen, presentations = (
                512,
                gate["distinct_records_seen"],
                gate["record_presentations"],
            )
        else:
            if (
                gate["worker_trial"] != settings["worker_trials"][rank]
                or gate["config_sha256"] != digest(config_path)
                or not gate["inherited_weights_unchanged"]
            ):
                raise ValueError("timed adaptation changed its declaration")
            updates = gate["updates"]
            name = "relay_" + settings["worker_trials"][rank]["name"]
            path = folder / ("mapper.pt" if rank == 1 else "adaptation.pt")
            if digest(path) != gate["checkpoint_sha256"]:
                raise ValueError("timed adaptation checkpoint changed")
            variants[name] = str(path) if rank == 1 else settings["initial_mapper"]
            if rank > 1:
                drafter_updates[name] = str(path)
            prepared, seen, presentations = (
                gate["prepared_distinct_examples"],
                gate["distinct_examples"],
                gate["presentations"],
            )
        if (
            prepared != 512
            or seen != min(512, 4 * updates)
            or presentations != 4 * updates
        ):
            raise ValueError("time-budget data accounting is inconsistent")
        timing_path = folder / (
            "budget-timing.jsonl" if rank == 0 else "training.jsonl"
        )
        history = [json.loads(line) for line in timing_path.read_text().splitlines()]
        values = [row["training_update_seconds"] for row in history]
        if (
            [row["step"] for row in history] != list(range(1, updates + 1))
            or values[-1] != actual
            or (values[-2] if len(values) > 1 else 0) != previous
            or any(a >= b for a, b in zip([0.0, *values[:-1]], values, strict=True))
            or any(value >= expected for value in values[:-1])
        ):
            raise ValueError(
                "raw update times do not prove first-crossing budget stopping"
            )
        destination = output / "fitting" / ("feature" if rank == 0 else f"rank{rank}")
        destination.mkdir(parents=True, exist_ok=False)
        for raw in folder.glob("*.json*"):
            shutil.copy2(raw, destination / raw.name)
        summaries.append(
            {
                "rank": rank,
                "method": name,
                "updates": updates,
                "prepared_records": prepared,
                "additional_records_seen": seen,
                "additional_presentations": presentations,
                "training_update_seconds": actual,
                "previous_update_seconds": previous,
                "budget_seconds": expected,
                "overshoot_seconds": actual - expected,
                "initial_mapper_training_charge_seconds": 0
                if rank == 0
                else budget["initial_mapper_charge_seconds"],
                "checkpoint": str(path),
                "checkpoint_sha256": digest(path),
                "fit_gate_sha256": digest(folder / gate_name),
                "fit_gate": gate,
            }
        )
    if args.pilot:
        del config["adaptation_pilot"]
        variants["relay_lora_zero"] = settings["initial_mapper"]
        drafter_updates["relay_lora_zero"] = str(fits / "rank2/zero-adaptation.pt")
        config["relay_probe"]["variants"] = variants
        config["relay_probe"]["drafter_updates"] = drafter_updates
        config["benchmark"]["methods"] = [
            "native_ar",
            "native_target_dflash",
            "optimized_source_reuse",
            *variants,
        ]
        path = output / "campaign-config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        subprocess.run(
            [
                python,
                "scripts/run_mapper_campaign.py",
                "--config",
                str(path),
                "--equal-methods",
                "relay_initial",
                "relay_lora_zero",
            ],
            env={**os.environ, "RELAYSPEC_OUTPUT": str(output / "evaluation")},
            check=True,
        )
    (output / "timed-budget-gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                **declaration,
                "pilot_decoding_passed": args.pilot,
                "fitting": summaries,
                "fit_wall_seconds": fit_wall_seconds,
                "total_seconds": time.perf_counter() - started,
                "variants": variants,
                "drafter_updates": drafter_updates,
                "scope": "Measured warm-training budgets with one-update overshoot recorded. Feature fitting starts "
                "from random initialization. CE and LoRA use the same initial mapper and are charged its measured "
                "training-update cost. Setup, teacher preparation, validation, export and shared cache construction "
                "are additional costs. Pilot decoding is a bounded correctness/resource check, not full-answer quality.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
