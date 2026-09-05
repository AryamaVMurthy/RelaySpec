"""Audit all small-data development quality rows and expose cap-length outputs."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from relayspec.ar_paper_evidence import load_scored, summarize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = Path(
        "configs/submission/scaling/campaign-small-data-quality-development.yaml"
    )
    provenance_path = Path(
        "configs/submission/scaling/small-data-quality.provenance.json"
    )
    config = yaml.safe_load(config_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    gate = json.loads((args.run / "completion-gate.json").read_text())
    analysis_path = args.run / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    if (
        digest(config_path) != provenance["configs_sha256"][config_path.name]
        or gate["status"] != "pass"
        or gate["requests"] != 128
        or gate["records"] != 1536
        or analysis["status"] != "complete"
        or any(
            digest(args.run / name) != sha
            for name, sha in analysis["raw_sha256"].items()
        )
    ):
        raise ValueError("quality campaign or scoring provenance is incomplete")
    rows = load_scored(args.run)
    if len(rows) != 1536 or {r["method"] for r in rows} != set(
        config["benchmark"]["methods"]
    ):
        raise ValueError("quality campaign has missing or unexpected method rows")
    for row in rows:
        expected = provenance["variants"].get(row["method"])
        if (
            expected
            and row.get("mapper_checkpoint_sha256") != expected["checkpoint_sha256"]
        ):
            raise ValueError("quality row used an undeclared mapper checkpoint")
    reference = summarize(rows)
    dense = summarize(rows, reference="relay_dense_n2048")
    if reference["requests"] != 128:
        raise ValueError("expected128 distinct paired quality requests")
    cap = config["generation"]["max_new_tokens"]
    cap_counts = {
        method: sum(r["output_tokens"] == cap for r in rows if r["method"] == method)
        for method in config["benchmark"]["methods"]
    }
    if any(r["output_tokens"] > cap for r in rows):
        raise ValueError("quality campaign output exceeds its declared cap")
    args.output.write_text(
        json.dumps(
            {
                "status": "complete",
                "run": str(args.run),
                "rows": len(rows),
                "config_sha256": digest(config_path),
                "provenance_sha256": digest(provenance_path),
                "completion_gate_sha256": digest(args.run / "completion-gate.json"),
                "analysis_sha256": digest(analysis_path),
                "scored_rows_sha256": digest(args.run / "math-scored.jsonl"),
                "raw_sha256": analysis["raw_sha256"],
                "against_ar": reference,
                "against_dense_n2048": dense,
                "output_cap": cap,
                "cap_length_output_counts": cap_counts,
                "scope": "128 exposed MATH development requests, nine frozen small-data mappers "
                "and three inherited references. Greedy2048-token outputs scored with the "
                "pinned math scorer. Paired request bootstrap95% intervals,10000 samples, "
                "seed1729. Cap-length outputs may include an EOS exactly at the cap, so "
                "these counts are not asserted to be exact truncation counts. Outcomes "
                "at the configured cap do not prove uncapped full-answer quality. "
                "Architecture/epoch comparisons are development evidence. Confirmation "
                "and reserve sets remain separate and unused.",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "requests": reference["requests"],
                "methods": len(cap_counts),
                "cap_length_output_counts": cap_counts,
            }
        )
    )


if __name__ == "__main__":
    main()
