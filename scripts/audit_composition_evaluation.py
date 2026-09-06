"""Reconstruct paired domain-composition decoding from fixed methods and requests."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from relayspec.ar_paper_evidence import load_scored, read_rows, summarize
from relayspec.cached_fit_evidence import digest
from relayspec.paired_accuracy import paired_accuracy_interval
from relayspec.quality_scoring import verify_saved_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=["pilot", "full"], required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--pilot-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pp = Path(
        "configs/submission/scaling/composition-small-v1/evaluation-bound-protocol.json"
    )
    protocol = json.loads(pp.read_text())
    inputs = {str(pp): digest(pp)}
    for path, sha in protocol["input_sha256"].items():
        assert digest(Path(path)) == sha
        inputs[path] = sha
    if args.stage == "full":
        assert args.pilot_audit is not None
        pilot = json.loads(args.pilot_audit.read_text())
        assert (
            pilot["status"] == "complete"
            and pilot["stage"] == "pilot"
            and pilot["protocol_sha256"] == digest(pp)
        )
        for path, sha in pilot["input_sha256"].items():
            assert digest(Path(path)) == sha
        inputs[str(args.pilot_audit)] = digest(args.pilot_audit)
    declaration = protocol["stages"][args.stage]
    cp = Path(declaration["config"])
    config = yaml.safe_load(cp.read_text())
    assert digest(cp) == declaration["config_sha256"]
    assert digest(Path(declaration["manifest"])) == declaration["manifest_sha256"]
    assert yaml.safe_load((args.run / "config.yaml").read_text()) == config
    ledger = json.loads(args.ledger.read_text())
    job = next(j for j in ledger["jobs"] if Path(j["local"]).name == args.run.name)
    assert job["stage"] == args.stage
    assert (args.run / "source-commit.txt").read_text().strip() == ledger[
        "source_commit"
    ]
    subprocess.run(
        [
            sys.executable,
            "scripts/check_run_complete.py",
            "--config",
            str(cp),
            "--output",
            str(args.run),
        ],
        check=True,
        capture_output=True,
    )
    campaign = json.loads((args.run / "campaign-gate.json").read_text())
    assert campaign["status"] == "pass" and campaign["methods"] == protocol["methods"]
    rows = read_rows(args.run)
    references = {
        r["problem_id"]: r.get("answer")
        for r in json.loads(Path(declaration["manifest"]).read_text())["records"]
        if r["benchmark"] in ["math500", "gsm8k"]
    }
    for row in rows:
        assert 0 < row["output_tokens"] <= config["generation"]["max_new_tokens"]
        if row["benchmark"] in ["math500", "gsm8k"]:
            assert str(row["reference_answer"]) == str(references[row["problem_id"]])
        if row["method"] in protocol["variants"]:
            assert (
                row["mapper_checkpoint_sha256"]
                == protocol["variants"][row["method"]]["checkpoint_sha256"]
            )
    scored = load_scored(args.run)
    verify_saved_scores(args.run, Path("/home/aryamavmurthy/work/RelaySpec"))
    score_lookup = {(r["problem_id"], r["method"]): r for r in scored}
    rows = [score_lookup.get((r["problem_id"], r["method"]), r) for r in rows]
    tasks = {}
    for task in protocol["declaration"]["tasks"]:
        task_rows = [r for r in rows if r["benchmark"] == task]
        pairs = {}
        for math_method in protocol["variants"]:
            if "_math_" not in math_method:
                continue
            mixed_method = math_method.replace("_math_", "_mixed_")
            pair_rows = [
                r for r in task_rows if r["method"] in [math_method, mixed_method]
            ]
            result = summarize(pair_rows, reference=math_method)
            if task in ["gsm8k", "math500"]:
                groups = {}
                for r in pair_rows:
                    groups.setdefault(r["problem_id"], {})[r["method"]] = r["correct"]
                result["conservative_paired_accuracy"] = paired_accuracy_interval(
                    [g[mixed_method] for g in groups.values()],
                    [g[math_method] for g in groups.values()],
                )
            pairs[math_method] = {"mixed_method": mixed_method, **result}
        tasks[task] = {
            "against_ar": summarize(task_rows),
            "composition_pairs": pairs,
            "cap_counts": {
                m: sum(
                    r["method"] == m
                    and r["output_tokens"] == config["generation"]["max_new_tokens"]
                    for r in task_rows
                )
                for m in protocol["methods"]
            },
        }
    for path in [
        cp,
        args.ledger,
        *sorted(args.run.glob("*.json")),
        *sorted(args.run.glob("*.jsonl")),
    ]:
        inputs[str(path)] = digest(path)
    output = {
        "status": "complete",
        "stage": args.stage,
        "protocol_sha256": digest(pp),
        "rows": len(rows),
        "tasks": tasks,
        "input_sha256": inputs,
        "scope": protocol["declaration"]["scope"],
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(args.stage, len(rows), "rows audited")


if __name__ == "__main__":
    main()
