"""Verify the common three-rate adaptation screen from its raw artifacts."""

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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = Path("configs/submission/scaling/drafter-adaptation-lr-screen.yaml")
    config = yaml.safe_load(config_path.read_text())
    settings = config["adaptation_pilot"]
    gate_path = args.run / "adaptation-screen-gate.json"
    gate = json.loads(gate_path.read_text())
    provenance_path = args.run / "screen-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if (
        gate["status"] != "pass"
        or gate["config_sha256"] != digest(config_path)
        or gate["provenance_sha256"] != digest(provenance_path)
        or provenance["prerequisite_gate_sha256"]
        != settings["prerequisite_gate"]["sha256"]
        or provenance["campaign_sha256"] != digest(args.run / "campaign-config.yaml")
    ):
        raise ValueError("adaptation screen provenance is incomplete")
    inputs = {
        str(p): digest(p)
        for p in [
            config_path,
            gate_path,
            provenance_path,
            args.run / "campaign-config.yaml",
        ]
    }
    for rank, trial in enumerate(settings["worker_trials"]):
        folder = args.run / "fitting" / f"rank{rank}"
        path = folder / "adaptation-fit-gate.json"
        fit = json.loads(path.read_text())
        history_path = folder / "training.jsonl"
        history = [json.loads(line) for line in history_path.read_text().splitlines()]
        if (
            fit != gate["fitting"][rank]
            or fit["worker_trial"] != trial
            or fit["updates"] != 128
            or fit["distinct_examples"] != 512
            or len(set(fit["ordered_record_files"])) != 512
            or [r["step"] for r in history] != list(range(1, 129))
            or fit["ordered_record_files"] != gate["fitting"][0]["ordered_record_files"]
        ):
            raise ValueError("screen fit does not reproduce the equal-data declaration")
        name = "relay_" + trial["name"]
        key = "drafter_update_sha256" if trial["lora_rank"] else "mapper_sha256"
        if provenance["variants"][name][key] != fit["checkpoint_sha256"]:
            raise ValueError("screen checkpoint differs from its fit gate")
        inputs.update(
            {str(path): digest(path), str(history_path): digest(history_path)}
        )
    for name, reused in settings["reused_variants"].items():
        if provenance["variants"][name] != reused:
            raise ValueError("reused comparison differs from the declaration")
    evaluation = args.run / "evaluation"
    completion = json.loads((evaluation / "completion-gate.json").read_text())
    analysis = json.loads((evaluation / "analysis.json").read_text())
    campaign = yaml.safe_load((args.run / "campaign-config.yaml").read_text())
    if (
        completion["status"] != "pass"
        or analysis["status"] != "complete"
        or yaml.safe_load((evaluation / "config.yaml").read_text()) != campaign
    ):
        raise ValueError("screen decoding configuration or completeness differs")
    for name in ["completion-gate.json", "analysis.json", "config.yaml"]:
        inputs[str(evaluation / name)] = digest(evaluation / name)
    for name, sha in analysis["raw_sha256"].items():
        if digest(evaluation / name) != sha:
            raise ValueError("raw screen decoding changed")
        inputs[str(evaluation / name)] = sha
    rows = read_rows(evaluation)
    if len(rows) != completion["records"] or {r["method"] for r in rows} != set(
        campaign["benchmark"]["methods"]
    ):
        raise ValueError("screen comparison omits declared methods")
    for row in rows:
        variant = provenance["variants"].get(row["method"])
        if variant and row.get("mapper_checkpoint_sha256") != variant["mapper_sha256"]:
            raise ValueError("screen used another mapper")
        if (
            variant
            and "drafter_update_sha256" in variant
            and row.get("drafter_checkpoint_sha256") != variant["drafter_update_sha256"]
        ):
            raise ValueError("screen used another adapted drafter")
    paired = summarize(rows, reference="relay_initial")
    if paired["requests"] != 16:
        raise ValueError("screen requires sixteen paired development requests")
    result = {
        "status": "complete",
        "input_sha256": inputs,
        "fitting": gate["fitting"],
        "against_initial": paired,
        "against_dense_endpoint": summarize(rows, reference="relay_dense_endpoint"),
        "scope": "Fixed equal-data screen, three learning rates per connector CE "
        "and rank32 drafter LoRA. 512 records and 128 updates per fit, with older "
        "high-rate fits reused. Sixteen exposed development requests at a 256-token "
        "cap. Descriptive paired request intervals, not fitting-seed uncertainty, "
        "full-answer quality, or matched-compute evidence.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {
                    field: v[field]
                    for field in [
                        "tokens_per_second",
                        "throughput_ratio",
                        "throughput_ci95",
                    ]
                }
                for k, v in paired["methods"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
