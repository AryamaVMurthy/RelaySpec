"""Audit equal-data adaptation timing, identity controls, and saved decoding."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from relayspec.ar_paper_evidence import read_rows, summarize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    settings = config["adaptation_pilot"]
    gate_path = args.run / "adaptation-pilot-gate.json"
    gate = json.loads(gate_path.read_text())
    if gate["status"] != "pass" or gate["config_sha256"] != digest(args.config):
        raise ValueError("adaptation calibration has not passed its declared pilot")
    inputs = {str(args.config): digest(args.config), str(gate_path): digest(gate_path)}
    fits = []
    for rank in range(4):
        folder = args.run / "fitting" / f"rank{rank}"
        fit_path = folder / "adaptation-fit-gate.json"
        history_path = folder / "training.jsonl"
        fit = json.loads(fit_path.read_text())
        history = [json.loads(line) for line in history_path.read_text().splitlines()]
        if (
            fit != gate["fitting"][rank]
            or fit["status"] != "pass"
            or fit["updates"] != settings["pilot_updates"]
            or fit["distinct_examples"] != settings["pilot_distinct_examples"]
            or len(set(fit["ordered_record_files"])) != fit["distinct_examples"]
            or [h["step"] for h in history] != list(range(1, fit["updates"] + 1))
            or history[-1]["presentations"] != fit["presentations"]
            or not fit["inherited_weights_unchanged"]
            or fit["direct_fusion_equivalence"]["status"] != "pass"
        ):
            raise ValueError("missing equal-data fit, history, or identity evidence")
        if rank and fit["ordered_record_files"] != fits[0]["ordered_record_files"]:
            raise ValueError("adaptation workers used different ordered records")
        inputs.update(
            {str(fit_path): digest(fit_path), str(history_path): digest(history_path)}
        )
        fits.append(fit)
    if fits[2]["export"]["checkpoint_sha256"] != fits[3]["export"]["checkpoint_sha256"]:
        raise ValueError("duplicate adapted weights differ")
    evaluation = args.run / "evaluation"
    completion = json.loads((evaluation / "completion-gate.json").read_text())
    analysis = json.loads((evaluation / "analysis.json").read_text())
    manifest = json.loads((evaluation / "mapper-campaign.json").read_text())
    for path in [
        evaluation / name
        for name in [
            "completion-gate.json",
            "analysis.json",
            "mapper-campaign.json",
            "config.yaml",
        ]
    ]:
        inputs[str(path)] = digest(path)
    if completion["status"] != "pass" or analysis["status"] != "complete":
        raise ValueError("adaptation decoding is incomplete")
    for name, sha in analysis["raw_sha256"].items():
        path = evaluation / name
        if digest(path) != sha:
            raise ValueError("raw adaptation decoding changed")
        inputs[str(path)] = sha
    rows = read_rows(evaluation)
    for row in rows:
        variant = manifest["variants"].get(row["method"])
        if variant and row.get("mapper_checkpoint_sha256") != variant["sha256"]:
            raise ValueError("decoding used another mapper")
        if (
            variant
            and "drafter_update" in variant
            and row.get("drafter_checkpoint_sha256")
            != variant["drafter_update"]["sha256"]
        ):
            raise ValueError("decoding used another drafter update")
    for method, rank in [
        ("relay_lora8", 1),
        ("relay_lora32", 2),
        ("relay_lora32_duplicate", 3),
    ]:
        if (
            manifest["variants"][method]["drafter_update"]["sha256"]
            != fits[rank]["export"]["checkpoint_sha256"]
        ):
            raise ValueError("decoded drafter differs from its verified export")
    for left, right in [
        ("relay_initial", "relay_lora_zero"),
        ("relay_lora32", "relay_lora32_duplicate"),
    ]:
        pairs = [
            {(r["problem_id"], r["repetition"]): r for r in rows if r["method"] == name}
            for name in (left, right)
        ]
        if pairs[0].keys() != pairs[1].keys() or not pairs[0]:
            raise ValueError("identity control requests differ")
        for key, row in pairs[0].items():
            if any(
                row[field] != pairs[1][key][field]
                for field in (
                    "output_hash",
                    "output_tokens",
                    "accepted_draft_lengths",
                    "proposal_lengths",
                )
            ):
                raise ValueError("identity control decoding differs")
    paired = summarize(rows, reference="relay_initial")
    if paired["requests"] != 8 or len(rows) != completion["records"]:
        raise ValueError("pilot is not the declared eight-request comparison")
    result = {
        "status": "complete",
        "input_sha256": inputs,
        "fitting": fits,
        "against_initial_mapper": paired,
        "scope": "Equal-data 512-record one-pass calibration, not matched-compute "
        "or final task-quality evidence. Fitting-loop time excludes model/teacher "
        "preparation, initial mapper fitting, shared feature-cache construction, "
        "checkpoint export and decoding. Preparation and total worker times are "
        "reported separately. Eight exposed requests, 128-token cap, greedy decoding. "
        "Direct fusion equality covers four teacher-forced records. Raw trainable "
        "state is serialized, but continuation equality still requires validation.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {
                    k: values[k]
                    for k in [
                        "tokens_per_second",
                        "throughput_ratio",
                        "throughput_ci95",
                    ]
                }
                for name, values in paired["methods"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
