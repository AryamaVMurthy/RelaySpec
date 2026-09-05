"""Score saved math text and summarize paired AR throughput without new GPU work."""

import argparse
import hashlib
import json
from pathlib import Path

from aggregate_results import build_math_scorer

from relayspec.ar_paper_evidence import read_rows, summarize


def analyze(directory, scorer_path):
    gate = json.loads((directory / "completion-gate.json").read_text())
    if gate["status"] != "pass":
        raise ValueError("run has not passed completeness checks")
    rows = read_rows(directory)
    if len(rows) != gate["records"]:
        raise ValueError("downloaded row count differs from completion gate")
    provenance_path = Path("reports/ar-revision-20260905/scorer-provenance.json")
    provenance = json.loads(provenance_path.read_text())
    for filename, expected in provenance["sha256"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"pinned scorer provenance changed: {filename}")
    scorer, scorer_name = build_math_scorer(scorer_path)
    cache = {}
    scored = []
    for row in rows:
        if row["benchmark"] in {"math500", "gsm8k"}:
            key = (row["completion"], str(row["reference_answer"]), row["benchmark"])
            if key not in cache:
                cache[key] = scorer(*key)
            row = {**row, **cache[key]}
            scored.append(row)
    score_lookup = {(r["problem_id"], r["repetition"], r["method"]): r for r in scored}
    rows = [
        score_lookup.get((r["problem_id"], r["repetition"], r["method"]), r)
        for r in rows
    ]
    (directory / "math-scored.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in scored)
    )
    tasks = {}
    for task in sorted({r["benchmark"] for r in rows}):
        subset = [r for r in rows if r["benchmark"] == task]
        tasks[task] = summarize(subset)
        for name, values in tasks[task]["methods"].items():
            selected = [r for r in subset if r["method"] == name]
            if all(
                "proposal_lengths" in r and "accepted_draft_lengths" in r
                for r in selected
            ):
                proposed = [v for r in selected for v in r["proposal_lengths"]]
                accepted = [v for r in selected for v in r["accepted_draft_lengths"]]
                if len(proposed) != len(accepted) or any(
                    not 0 <= a <= p for a, p in zip(accepted, proposed, strict=True)
                ):
                    raise ValueError("invalid acceptance counts")
                if sum(proposed):
                    values["proposed_token_acceptance"] = sum(accepted) / sum(proposed)
        native = next(
            (m for m in tasks[task]["methods"] if m.startswith("native_target_")), None
        )
        if native:
            tasks[task]["native_reference"] = summarize(subset, reference=native)
    result = {
        "status": "complete",
        "tasks": tasks,
        "scorer": scorer_name,
        "scorer_provenance_sha256": hashlib.sha256(
            provenance_path.read_bytes()
        ).hexdigest(),
        "scorer_files": {
            str(p.relative_to(scorer_path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in scorer_path.rglob("*.py")
        },
        "raw_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.glob("benchmark-rank*.jsonl")
        },
        "scope": "Paired throughput and 95% bootstrap intervals. Conversations are resampled as units. Math scores use saved outputs. Code and conversation quality require their own evaluators. Acceptance counts all verified draft positions before EOS/cap trimming.",
    }
    (directory / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--scorer", type=Path, default=Path("vendor/qwen-math/evaluation")
    )
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "directory": str(args.directory),
                "status": analyze(args.directory, args.scorer)["status"],
            }
        )
    )


if __name__ == "__main__":
    main()
