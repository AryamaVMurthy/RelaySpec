"""Reproduce the prospective PARD gate and runtime-local throughput comparison."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from relayspec.ar_paper_evidence import load_scored, read_rows, summarize
from relayspec.paired_accuracy import paired_accuracy_interval


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/submission/baselines/pard-campaign.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    ledger = json.loads(args.ledger.read_text())
    if (
        (run / "source-commit.txt").read_text().strip() != ledger["source_commit"]
        or len([j for j in ledger["jobs"] if Path(j["local"]).name == run.name]) != 1
        or args.config.read_bytes() != (run / "campaign-config.json").read_bytes()
    ):
        raise ValueError("PARD campaign source/job/config differs from declaration")
    config = json.loads(args.config.read_text())
    for path, sha in config.get("prerequisites", {}).items():
        prerequisite = Path(path)
        if (
            digest(prerequisite) != sha
            or json.loads(prerequisite.read_text())["status"] != "complete"
        ):
            raise ValueError("PARD longer-output prerequisite changed")
    protocol_path = Path(config["verification_protocol"])
    protocol = json.loads(protocol_path.read_text())
    source_config = Path(config["source_config"])
    setup = json.loads((run / "setup-gate.json").read_text())
    if (
        digest(protocol_path) != config["verification_protocol_sha256"]
        or any(digest(Path(p)) != sha for p, sha in protocol["input_sha256"].items())
        or digest(source_config) != config["source_config_sha256"]
        or setup["status"] != "pass"
        or setup["config_sha256"] != digest(source_config)
    ):
        raise ValueError("PARD prospective evidence or setup changed")
    raw_files = [*run.glob("benchmark-rank*.jsonl"), *run.glob("campaign-rank*.json")]
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for path in raw_files:
            (directory / path.name).symlink_to(path)
        env = os.environ.copy()
        env["RELAYSPEC_OUTPUT"] = str(directory)
        subprocess.run(
            [
                sys.executable,
                "scripts/check_pard_campaign.py",
                "--config",
                str(args.config),
            ],
            env=env,
            check=True,
            capture_output=True,
        )
        gate = json.loads((directory / "completion-gate.json").read_text())
        if gate != json.loads((run / "completion-gate.json").read_text()):
            raise ValueError("PARD completion gate does not reproduce")
    rows = read_rows(run)
    quality = None
    quality_inputs = []
    if config.get("phase") in {"quality_pilot", "quality_full"}:
        analysis_path = run / "analysis.json"
        analysis = json.loads(analysis_path.read_text())
        scorer_path = Path("reports/ar-revision-20260905/scorer-provenance.json")
        if (
            analysis["status"] != "complete"
            or analysis["scorer_provenance_sha256"] != digest(scorer_path)
            or analysis["raw_sha256"]
            != {p.name: digest(p) for p in run.glob("benchmark-rank*.jsonl")}
        ):
            raise ValueError(
                "PARD quality analysis does not bind the pinned scorer and raw outputs"
            )
        rows = load_scored(run)
        quality_inputs = [analysis_path, run / "math-scored.jsonl", scorer_path]
        quality = {
            "phase": config["phase"],
            "requests": config["requests"],
            "token_cap": config["max_new_tokens"],
            "methods": {
                name: {
                    "correct_count": sum(
                        bool(r["correct"]) for r in rows if r["method"] == name
                    ),
                    "cap_count": sum(
                        r["output_tokens"] == config["max_new_tokens"]
                        for r in rows
                        if r["method"] == name
                    ),
                    "eos_at_cap_count": sum(
                        r["output_tokens"] == config["max_new_tokens"]
                        and r["output_ids"][-1] == 151645
                        for r in rows
                        if r["method"] == name
                    ),
                }
                for name in config["methods"]
            },
            "scope": "Capped exposed development outcomes. Paired bootstrap intervals are descriptive, especially for small samples and sparse discordances. This does not establish the final one-percentage-point noninferiority requirement or untouched confirmation.",
        }
        paired = {}
        for row in rows:
            paired.setdefault(row["problem_id"], {})[row["method"]] = row["correct"]
        quality["conservative_paired_accuracy"] = paired_accuracy_interval(
            [g["pard"] for g in paired.values()],
            [g["native_ar"] for g in paired.values()],
        )
        quality_inputs.append(Path("src/relayspec/paired_accuracy.py"))
    result = {
        "status": "complete",
        "input_sha256": {
            str(p): digest(p)
            for p in [
                args.config,
                args.ledger,
                protocol_path,
                source_config,
                run / "source-commit.txt",
                run / "setup-gate.json",
                run / "campaign-config.json",
                run / "completion-gate.json",
                *raw_files,
                *quality_inputs,
            ]
        },
        "comparison": summarize(rows),
        "ar_exact_count": gate["ar_exact_count"],
        "verification_replica_seconds": gate["verification_replica_seconds"],
        "original_exact_ar_gates": protocol["original_exact_ar_gates"],
        "scope": config["scope"],
    }
    if quality is not None:
        result["capped_quality"] = quality
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["comparison"]))


if __name__ == "__main__":
    main()
