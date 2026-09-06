"""Promote only an audited EAGLE compatibility pilot to equal-data calibration."""

import argparse
import json
from pathlib import Path

import yaml

from relayspec.cached_fit_evidence import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pilot = json.loads(args.pilot_result.read_text())
    if (
        pilot.get("status") != "complete"
        or pilot.get("family") != "eagle3"
        or pilot.get("stage") != "compatibility_pilot"
    ):
        raise ValueError(
            "equal-data EAGLE calibration requires the audited compatibility pilot"
        )
    for name, sha in pilot["input_sha256"].items():
        if digest(Path(name)) != sha:
            raise ValueError("EAGLE compatibility evidence changed")
    config_path = Path("configs/submission/scaling/eagle3-adaptation-pilot.yaml")
    ledger_path = Path(
        "reports/mapper-scaling-20260905/eagle3-adaptation-pilot/jobs.json"
    )
    if pilot["input_sha256"].get(str(config_path)) != digest(config_path) or pilot[
        "input_sha256"
    ].get(str(ledger_path)) != digest(ledger_path):
        raise ValueError(
            "EAGLE pilot audit is not bound to its committed configuration/ledger"
        )
    ledger = json.loads(ledger_path.read_text())
    if len(ledger["jobs"]) != 1:
        raise ValueError("EAGLE pilot ledger must identify one exact compatibility run")
    job = ledger["jobs"][0]
    gate_paths = [
        Path(name)
        for name in pilot["input_sha256"]
        if Path(name).name == "adaptation-pilot-gate.json"
    ]
    if len(gate_paths) != 1 or gate_paths[0].parent.name != Path(job["local"]).name:
        raise ValueError("ambiguous EAGLE compatibility gate")
    gate = json.loads(gate_paths[0].read_text())
    config = yaml.safe_load(config_path.read_text())
    if gate["config_sha256"] != digest(config_path):
        raise ValueError("EAGLE compatibility gate uses a different configuration")
    config["run_name"] = "eagle3-small-drafter-adaptation-512-calibration"
    settings = config["adaptation_pilot"]
    settings.update(
        diagnostics=False,
        pilot_distinct_examples=512,
        pilot_updates=128,
        prepared_records_device="cpu",
        prerequisite_gate={
            "path": str(Path(job["remote"]) / "adaptation-pilot-gate.json"),
            "sha256": digest(gate_paths[0]),
        },
        scope="EAGLE-3 one-step shifted teacher-forced equal-data calibration:512 cached examples,128 batch-four updates, connector CE and frozen-mapper LoRA8/32 with duplicate rank32. Charge initial512-example/128-update feature mapper, cache construction, target-label preparation, optimizer work and export separately. Not multi-step TTT, matched-time superiority or final quality. Requires the source-bound compatibility pilot before any GPU fitting.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "status": "declared_after_audited_pilot",
                "pilot_result_sha256": digest(args.pilot_result),
                "pilot_gate_sha256": digest(gate_paths[0]),
                "config_sha256": digest(args.output),
                "distinct_examples": 512,
                "updates": 128,
                "scope": "Equal-data calibration declared, not executed. Four GPUs,540-second process limit and ten-minute Slurm ceiling. Large-data scaling remains paused.",
            },
            indent=2,
        )
        + "\n"
    )
    print("Declared EAGLE512 equal-data calibration from its audited pilot")


if __name__ == "__main__":
    main()
