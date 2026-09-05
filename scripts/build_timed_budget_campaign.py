"""Declare one paired development campaign after all three timed fits pass."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    gates = {}
    for run in args.runs:
        path = run / "timed-budget-gate.json"
        gate = json.loads(path.read_text())
        multiplier = gate["budget"]["multiplier"]
        if multiplier in gates or gate["status"] != "pass":
            raise ValueError("missing or duplicated completed timed budget")
        gates[multiplier] = (run, gate, digest(path))
    if set(gates) != {0.25, 1.0, 4.0}:
        raise ValueError("all three declared budgets must finish before decoding")
    protocol_path = Path("configs/submission/scaling/adaptation-budget-protocol.json")
    if not gates[0.25][1]["pilot_decoding_passed"]:
        raise ValueError("timed checkpoint pilot did not pass decoding")
    config_path = Path(
        "configs/submission/scaling/drafter-adaptation-512-calibration.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    settings = config.pop("adaptation_pilot")
    initial = settings["initial_mapper"]
    variants, updates = {"relay_initial": initial}, {}
    evidence = {"relay_initial": {"mapper_sha256": settings["initial_mapper_sha256"]}}
    for multiplier, (run, gate, sha) in sorted(gates.items()):
        if len(gate["fitting"]) != 4 or sorted(r["rank"] for r in gate["fitting"]) != [
            0,
            1,
            2,
            3,
        ]:
            raise ValueError("timed fitting is missing a declared worker")
        if gate["protocol_sha256"] != digest(protocol_path) or (
            multiplier != 0.25 and gate["pilot_gate_sha256"] != gates[0.25][2]
        ):
            raise ValueError("time-budget fits use different protocols or pilots")
        for record in gate["fitting"]:
            rank = record["rank"]
            path = (
                run
                / "fitting"
                / (
                    "feature/fit-complete.json"
                    if rank == 0
                    else f"rank{rank}/adaptation-fit-gate.json"
                )
            )
            if (
                digest(path) != record["fit_gate_sha256"]
                or json.loads(path.read_text()) != record["fit_gate"]
            ):
                raise ValueError("timed fit evidence changed")
            if (
                rank
                and record["fit_gate"]["initial_mapper_sha256"]
                != settings["initial_mapper_sha256"]
            ):
                raise ValueError("adaptation budgets used another initial mapper")
            alias = f"relay_budget{round(multiplier * 100):03d}_" + record[
                "method"
            ].removeprefix("relay_")
            variants[alias] = record["checkpoint"] if rank < 2 else initial
            entry = {
                "multiplier": multiplier,
                "rank": rank,
                "updates": record["updates"],
                "fit_gate_sha256": record["fit_gate_sha256"],
                "budget_gate_sha256": sha,
                "mapper_sha256": record["checkpoint_sha256"]
                if rank < 2
                else settings["initial_mapper_sha256"],
            }
            if rank > 1:
                updates[alias] = record["checkpoint"]
                entry["drafter_update_sha256"] = record["checkpoint_sha256"]
            evidence[alias] = entry
    reference_path = Path("configs/submission/scaling/baseline-time-calibration.json")
    reference = json.loads(reference_path.read_text())["trials"][1][
        "reference_checkpoint"
    ]
    variants["relay_dense8192_reference"] = reference["path"]
    evidence["relay_dense8192_reference"] = {"mapper_sha256": reference["sha256"]}
    if args.check_files:
        for name, path in variants.items():
            if digest(Path(path)) != evidence[name]["mapper_sha256"]:
                raise ValueError("decoded mapper checkpoint differs from its fit")
        for name, path in updates.items():
            if digest(Path(path)) != evidence[name]["drafter_update_sha256"]:
                raise ValueError("decoded LoRA checkpoint differs from its fit")
    config["run_name"] = "matched-warm-training-budget-development"
    config["benchmark"]["max_prompts"] = 16
    config["generation"]["max_new_tokens"] = 256
    config["relay_probe"]["variants"] = variants
    config["relay_probe"]["drafter_updates"] = updates
    config["benchmark"]["methods"] = [
        "native_ar",
        "native_target_dflash",
        "optimized_source_reuse",
        *variants,
    ]
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "status": "declared_after_all_timed_fit_gates",
                "config_sha256": digest(args.output),
                "protocol_sha256": digest(protocol_path),
                "template_sha256": digest(config_path),
                "reference_declaration_sha256": digest(reference_path),
                "variants": evidence,
                "scope": "All twelve timed checkpoints plus initial and existing dense8192 references "
                "and three inherited controls. Sixteen paired exposed development requests, 256-token "
                "cap. Candidate set fixed by measured budgets before decoding. No final task-quality "
                "or untouched-confirmation claim. All fit pools remain at 512 examples.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Declared {len(variants)} mapper variants and three inherited controls")


if __name__ == "__main__":
    main()
