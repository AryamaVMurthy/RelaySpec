"""Audit input-only task strata over all frozen small-data mapper quality rows."""

import argparse
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from relayspec.ar_paper_evidence import load_scored, summarize
from relayspec.paired_accuracy import paired_accuracy_interval
from relayspec.task_complexity import input_strata


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = Path("configs/submission/scaling/task-complexity-development.json")
    protocol = json.loads(protocol_path.read_text())
    run = (args.raw_root / protocol["run"]).resolve()
    registry_path = Path(
        "reports/mapper-scaling-20260905/small-data-quality-results.json"
    )
    registry = json.loads(registry_path.read_text())
    config_path = Path(
        "configs/submission/scaling/campaign-small-data-quality-development.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    manifest_path = Path(protocol["manifest"])
    if (
        digest(manifest_path) != protocol["manifest_sha256"]
        or config["benchmark"]["manifest_path"] != str(manifest_path)
        or yaml.safe_load((run / "config.yaml").read_text()) != config
    ):
        raise ValueError("task labels or recorded campaign configuration changed")
    with tempfile.TemporaryDirectory() as temporary:
        expected = Path(temporary) / "quality.json"
        subprocess.run(
            [
                sys.executable,
                "scripts/summarize_small_quality.py",
                "--run",
                str(run),
                "--output",
                str(expected),
            ],
            check=True,
            capture_output=True,
        )
        if json.loads(expected.read_text()) != registry:
            raise ValueError("parent quality registry does not reproduce")
    scorer = Path("reports/ar-revision-20260905/scorer-provenance.json")
    analysis = json.loads((run / "analysis.json").read_text())
    if analysis["scorer_provenance_sha256"] != digest(scorer):
        raise ValueError("task scoring used a different pinned scorer")
    manifest = json.loads(manifest_path.read_text())["records"]
    selected = [
        r for r in manifest if r["benchmark"] in config["benchmark"]["benchmarks"]
    ]
    random.Random(config["seed"]).shuffle(selected)
    selected = selected[: config["benchmark"]["max_prompts"]]
    metadata = {r["problem_id"]: r for r in selected}
    rows = load_scored(run)
    paired = {}
    for row in rows:
        if row["repetition"] != 0:
            raise ValueError(
                "complexity accuracy requires one fixed realization per request"
            )
        group = paired.setdefault(row["problem_id"], {})
        if row["method"] in group:
            raise ValueError("duplicate method/request")
        group[row["method"]] = row
    if set(paired) != set(metadata):
        raise ValueError("observed request identities differ from declared evaluation")
    methods = set(config["benchmark"]["methods"])
    assignments = {}
    for pid, group in paired.items():
        if (
            set(group) != methods
            or len({r["input_tokens"] for r in group.values()}) != 1
        ):
            raise ValueError("methods are missing or have different input lengths")
        assignments[pid] = input_strata(
            metadata[pid], group["native_ar"]["input_tokens"], protocol
        )
    labels = {
        "difficulty": list(protocol["difficulty"]),
        "subject": sorted(
            {r["metadata"]["subject"] for r in manifest if r["benchmark"] == "math500"}
        ),
        "input_length": list(protocol["input_token_bins"]),
    }
    strata = {}
    for axis, names in labels.items():
        strata[axis] = {}
        for name in names:
            ids = sorted(
                pid
                for pid, assignment in assignments.items()
                if assignment[axis] == name
            )
            if not ids:
                strata[axis][name] = {"requests": 0, "status": "empty"}
                continue
            subset = [row for pid in ids for row in paired[pid].values()]
            comparisons = {
                ref: summarize(subset, reference=ref) for ref in protocol["references"]
            }
            counts = {}
            for method in sorted(methods):
                actual = [paired[pid][method] for pid in ids]
                recorded = [
                    ("proposal_lengths" in r, "accepted_draft_lengths" in r)
                    for r in actual
                ]
                if len(set(recorded)) != 1 or recorded[0][0] != recorded[0][1]:
                    raise ValueError("inconsistent proposed/accepted token coverage")
                proposed = (
                    sum(sum(r["proposal_lengths"]) for r in actual)
                    if recorded[0][0]
                    else None
                )
                accepted = (
                    sum(sum(r["accepted_draft_lengths"]) for r in actual)
                    if recorded[0][0]
                    else None
                )
                counts[method] = {
                    "correct": sum(bool(r["correct"]) for r in actual),
                    "cap_hits": sum(
                        r["output_tokens"] == registry["output_cap"] for r in actual
                    ),
                    "proposed_draft_tokens": proposed,
                    "accepted_draft_tokens": accepted,
                    "accepted_fraction": accepted / proposed if proposed else None,
                    "paired_accuracy_against_ar": paired_accuracy_interval(
                        [r["correct"] for r in actual],
                        [paired[pid]["native_ar"]["correct"] for pid in ids],
                    ),
                }
            strata[axis][name] = {
                "status": "complete",
                "requests": len(ids),
                "problem_ids": ids,
                "comparisons": comparisons,
                "counts": counts,
            }
        if sum(g["requests"] for g in strata[axis].values()) != len(paired):
            raise ValueError("stratum coverage changed")
    inputs = [
        protocol_path,
        manifest_path,
        config_path,
        registry_path,
        scorer,
        run / "analysis.json",
        run / "math-scored.jsonl",
        run / "completion-gate.json",
        *sorted(run.glob("benchmark-rank*.jsonl")),
        Path("src/relayspec/task_complexity.py"),
        Path("src/relayspec/paired_accuracy.py"),
    ]
    result = {
        "status": "complete",
        "requests": len(paired),
        "methods": sorted(methods),
        "input_sha256": {str(p): digest(p) for p in inputs},
        "assignments": assignments,
        "strata": strata,
        "scope": protocol["scope"] + " " + protocol["uncertainty"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                axis: {name: r["requests"] for name, r in groups.items()}
                for axis, groups in strata.items()
            }
        )
    )


if __name__ == "__main__":
    main()
