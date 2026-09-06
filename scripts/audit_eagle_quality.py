"""Recheck EAGLE quality requests, checkpoint identities, pinned scoring and paired outcomes."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from relayspec.ar_paper_evidence import load_scored, summarize
from relayspec.paired_accuracy import paired_accuracy_interval
from relayspec.quality_scoring import verify_saved_scores


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage", choices=["pilot", "full"], required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--scoring-repo", type=Path, required=True)
    parser.add_argument("--pilot-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    ledger = json.loads(args.ledger.read_text())
    stages = (
        ["pilot"]
        if args.stage == "pilot"
        else (["full"] if "full" in protocol["stages"] else ["full0", "full1"])
    )
    if len(args.runs) != len(stages):
        raise ValueError("quality audit needs exactly its declared shards")
    inputs = {
        str(p): digest(p)
        for p in [args.protocol, args.ledger, Path("src/relayspec/paired_accuracy.py")]
    }
    for relative in [
        "reports/ar-revision-20260905/scorer-provenance.json",
        "scripts/analyze_controlled_run.py",
        "src/relayspec/ar_paper_evidence.py",
    ]:
        path = args.scoring_repo / relative
        inputs[str(path)] = digest(path)
    for name, sha in protocol["input_sha256"].items():
        if digest(Path(name)) != sha:
            raise ValueError("quality selection evidence changed")
        inputs[name] = sha
    if args.stage == "full":
        if not args.pilot_result:
            raise ValueError("full quality requires its audited long-output pilot")
        pilot = json.loads(args.pilot_result.read_text())
        if (
            pilot.get("status") != "complete"
            or pilot["stage"] != "pilot"
            or pilot["protocol_sha256"] != digest(args.protocol)
        ):
            raise ValueError("quality pilot did not pass this exact declaration")
        for name, sha in pilot["input_sha256"].items():
            if digest(Path(name)) != sha:
                raise ValueError("quality pilot evidence changed")
            inputs[name] = sha
        inputs[str(args.pilot_result)] = digest(args.pilot_result)
    rows = []
    for stage, run in zip(stages, args.runs, strict=True):
        declared = protocol["stages"][stage]
        config_path, manifest_path = (
            Path(declared["config"]),
            Path(declared["manifest"]),
        )
        config = yaml.safe_load(config_path.read_text())
        job = next(j for j in ledger["jobs"] if Path(j["local"]).name == run.name)
        if (
            digest(config_path) != declared["config_sha256"]
            or digest(manifest_path) != declared["manifest_sha256"]
            or yaml.safe_load((run / "config.yaml").read_text()) != config
            or config["benchmark"]["methods"] != protocol["methods"]
            or config["generation"]["max_new_tokens"] != protocol["output_cap"]
            or job["stage"] != stage
            or (run / "source-commit.txt").read_text().strip()
            != ledger["source_commit"]
        ):
            raise ValueError("quality run differs from committed declaration/source")
        gate = json.loads((run / "completion-gate.json").read_text())
        campaign = json.loads((run / "campaign-gate.json").read_text())
        if (
            gate.get("status") != "pass"
            or campaign.get("status") != "pass"
            or campaign["methods"] != protocol["methods"]
        ):
            raise ValueError("quality run did not pass completeness")
        expected = {
            (pid, 0, method)
            for pid in declared["problem_ids"]
            for method in protocol["methods"]
        }
        shard = load_scored(run)
        if (
            len(shard) != len(expected)
            or {(r["problem_id"], r["repetition"], r["method"]) for r in shard}
            != expected
        ):
            raise ValueError(
                "quality rows lack the exact declared request/method pairs"
            )
        if gate["requests"] != declared["requests"] or gate["records"] != len(expected):
            raise ValueError("quality completion miscounts requests")
        references = {
            r["problem_id"]: r["answer"]
            for r in json.loads(manifest_path.read_text())["records"]
        }
        for row in shard:
            variant = protocol["variants"].get(row["method"])
            if (
                row["benchmark"] != protocol.get("benchmark", "math500")
                or not isinstance(row["correct"], bool)
                or not 0 < row["output_tokens"] <= protocol["output_cap"]
                or str(row["reference_answer"]) != str(references[row["problem_id"]])
                or (
                    variant
                    and row.get("mapper_checkpoint_sha256")
                    != variant["checkpoint_sha256"]
                )
            ):
                raise ValueError(
                    "quality row has an undeclared checkpoint, answer or cap"
                )
        # Re-score saved text; do not trust an unbound correctness column.
        verify_saved_scores(run, args.scoring_repo)
        rows.extend(shard)
        for path in [
            config_path,
            manifest_path,
            run / "config.yaml",
            run / "source-commit.txt",
            run / "completion-gate.json",
            run / "campaign-gate.json",
            run / "analysis.json",
            run / "math-scored.jsonl",
            *sorted(run.glob("benchmark-rank*.jsonl")),
        ]:
            inputs[str(path)] = digest(path)
    paired = {}
    for row in rows:
        group = paired.setdefault(row["problem_id"], {})
        if row["method"] in group:
            raise ValueError("quality shards overlap")
        group[row["method"]] = row["correct"]
    comparisons = {}
    for reference in ["native_ar", protocol["dense_reference"]]:
        comparisons[reference] = {
            "paired_throughput": summarize(rows, reference=reference),
            "conservative_paired_accuracy": {
                method: paired_accuracy_interval(
                    [g[method] for g in paired.values()],
                    [g[reference] for g in paired.values()],
                )
                for method in protocol["methods"]
                if method != reference
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "complete",
                "stage": args.stage,
                "protocol_sha256": digest(args.protocol),
                "input_sha256": inputs,
                "requests": len(paired),
                "rows": len(rows),
                "comparisons": comparisons,
                "methods": {
                    method: {
                        "correct_count": sum(
                            r["correct"] for r in rows if r["method"] == method
                        ),
                        "cap_length_outputs": sum(
                            r["output_tokens"] == protocol["output_cap"]
                            for r in rows
                            if r["method"] == method
                        ),
                    }
                    for method in protocol["methods"]
                },
                "scope": protocol["scope"]
                if "full" in protocol["stages"]
                else "Source-bound exposed development outcomes at2048-token cap, with pinned scorer replay. Cap-length output is not asserted to exclude EOS at the cap. All candidates and negative outcomes are retained. Conservative paired intervals are individual comparisons under IID request-pair assumptions, not simultaneous selection guarantees. No untouched-confirmation or uncapped-quality claim.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"requests": len(paired), "rows": len(rows), "stage": args.stage}))


if __name__ == "__main__":
    main()
