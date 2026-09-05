"""Freeze measured adaptation budgets after rate selection and resume checks."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from relayspec.ar_paper_evidence import read_rows, summarize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {}
    registries = {}
    for name, script, relative in [
        (
            "baseline-time-calibration",
            "audit_baseline_timing.py",
            "baseline-time-calibration/run-27851",
        ),
        (
            "adaptation-lr-screen",
            "audit_adaptation_screen.py",
            "adaptation-lr-screen/run-27849",
        ),
    ]:
        registry = Path("reports/external-baselines-20260906") / f"{name}.json"
        result = json.loads(registry.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            expected = Path(tmp) / "result.json"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/" + script,
                    "--run",
                    str(args.raw_root / "reports/mapper-scaling-20260905" / relative),
                    "--output",
                    str(expected),
                ],
                check=True,
                capture_output=True,
            )
            if json.loads(expected.read_text()) != result:
                raise ValueError("budget source does not reproduce from raw artifacts")
        inputs[str(registry)] = digest(registry)
        registries[name] = result
    run = (
        args.raw_root
        / "reports/mapper-scaling-20260905/adaptation-continuation/run-27852"
    )
    gate_path = run / "adaptation-continuation-gate.json"
    gate = json.loads(gate_path.read_text())
    config_path = Path(
        "configs/submission/scaling/drafter-adaptation-continuation.yaml"
    )
    if (
        gate["status"] != "pass"
        or gate["config_sha256"] != digest(config_path)
        or not gate["decoding_bit_identical"]
        or gate["pairs"]
        != [
            {"workers": [0, 1], "weights_and_optimizer_bit_identical": True},
            {"workers": [2, 3], "weights_and_optimizer_bit_identical": True},
        ]
    ):
        raise ValueError("budget continuation prerequisite has not passed")
    inputs.update(
        {str(gate_path): digest(gate_path), str(config_path): digest(config_path)}
    )
    for rank, fit in enumerate(gate["fitting"]):
        path = run / "fitting" / f"rank{rank}" / "adaptation-fit-gate.json"
        history_path = path.parent / "training.jsonl"
        history = [json.loads(line) for line in history_path.read_text().splitlines()]
        start = 128 if rank % 2 else 0
        if (
            json.loads(path.read_text()) != fit
            or fit["start_step"] != start
            or fit["updates"] != 256
            or fit["updates_this_job"] != 256 - start
            or [r["step"] for r in history] != list(range(start + 1, 257))
        ):
            raise ValueError("continuation step accounting is inconsistent")
        inputs.update(
            {str(path): digest(path), str(history_path): digest(history_path)}
        )
    evaluation = run / "evaluation"
    analysis_path = evaluation / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    if analysis["status"] != "complete":
        raise ValueError("continuation decoding was not collected completely")
    inputs[str(analysis_path)] = digest(analysis_path)
    for name, sha in analysis["raw_sha256"].items():
        path = evaluation / name
        if digest(path) != sha:
            raise ValueError("continuation raw decoding changed")
        inputs[str(path)] = sha
    rows = read_rows(evaluation)
    for left, right in [
        ("relay_ce_full", "relay_ce_resumed"),
        ("relay_lora_full", "relay_lora_resumed"),
    ]:
        pair = [
            {
                (r["problem_id"], r["repetition"]): r
                for r in rows
                if r["method"] == method
            }
            for method in (left, right)
        ]
        if len(pair[0]) != 8 or pair[0].keys() != pair[1].keys():
            raise ValueError("continuation decoding pair is incomplete")
        if any(
            any(
                row[k] != pair[1][key][k]
                for k in (
                    "output_hash",
                    "output_tokens",
                    "accepted_draft_lengths",
                    "proposal_lengths",
                )
            )
            for key, row in pair[0].items()
        ):
            raise ValueError("continued decoding differs from uninterrupted decoding")
    screen = registries["adaptation-lr-screen"]["against_initial"]["methods"]
    selection = {}
    for family in ("ce", "lora32"):
        rates = {
            f"relay_{family}_lr2e5": 0.00002,
            f"relay_{family}_lr6e5": 0.00006,
            f"relay_{family}_lr2e4": 0.0002,
        }
        selected = max(rates, key=lambda name: screen[name]["tokens_per_second"])
        selection[family] = {
            "method": selected,
            "learning_rate": rates[selected],
            "selection": "highest throughput point in the fixed three-rate development screen",
        }
    result = {
        "status": "declared_requires_timed_checkpoint_pilot",
        "input_sha256": inputs,
        "distinct_training_pool_records": 512,
        "batch_size": 4,
        "selected_rates": selection,
        "warm_training_budgets": registries["baseline-time-calibration"][
            "warm_training_budgets"
        ],
        "continuation": {
            "status": "pass",
            "pairs": gate["pairs"],
            "decoding": summarize(rows, reference="relay_initial"),
        },
        "execution_requirements": [
            "Implement measured-time checkpointing and run a quarter-budget pilot under ten minutes before the full trajectory.",
            "Stop at the first completed optimizer update reaching each threshold, record actual time and at-most-one-update overshoot.",
            "Exclude checkpoint I/O from warm optimizer time and charge it in total worker and allocation time.",
            "Subtract the measured initial-mapper update cost from each CE/LoRA warm budget, and report its full setup/export cost separately.",
            "Report additional records actually seen and presentations, including initial mapper fitting, rather than relabeling repeated passes as new data.",
            "Record model load, teacher preparation, shared feature-cache construction/storage, validation, export, fit, decoding and search costs.",
            "Report fresh-setup budget infeasibility explicitly where required preparation alone exceeds that budget.",
            "Compare fitted feature-map references, connector CE and frozen-connector LoRA in one declared decoding campaign after fitting.",
            "Use development prompts only. Final confirmation remains unused until candidate selection is frozen.",
        ],
        "large_data_scaling": "paused",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
