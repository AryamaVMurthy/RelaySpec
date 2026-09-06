"""Rebuild the SD-square declaration and decision gate from raw fitting/decoding."""

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


def audit_shard(args, run, index):
    run = run.resolve()
    ledger = json.loads(args.ledger.read_text())
    if (run / "source-commit.txt").read_text().strip() != ledger[
        "source_commit"
    ] or len([j for j in ledger["jobs"] if Path(j["local"]).name == run.name]) != 1:
        raise ValueError("SD-square decoding source/job differs from launch record")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        config_path = directory / "campaign-config.json"
        builder = (
            [
                sys.executable,
                "scripts/build_sd_square_quality_campaign.py",
                *(["--full"] if getattr(args, "quality_full", False) else []),
                "--shard-index",
                str(index),
                "--output",
                str(config_path),
            ]
            if getattr(args, "quality", False) or getattr(args, "quality_full", False)
            else [
                sys.executable,
                "scripts/build_sd_square_epoch_campaign.py",
                "--run",
                str(args.epoch_run),
                "--shard-index",
                str(index),
                "--output",
                str(config_path),
            ]
            if args.epoch_run is not None
            else [
                sys.executable,
                "scripts/build_sd_square_campaign.py",
                "--shard-index",
                str(index),
                "--runs",
                *map(str, args.fits),
                "--output",
                str(config_path),
            ]
        )
        subprocess.run(
            builder,
            check=True,
            capture_output=True,
        )
        if config_path.read_bytes() != (run / config_path.name).read_bytes():
            raise ValueError("SD-square campaign does not reproduce from fitted inputs")
        for path in [
            *run.glob("benchmark-rank*.jsonl"),
            *run.glob("campaign-rank*.json"),
        ]:
            (directory / path.name).symlink_to(path)
        env = os.environ.copy()
        env["RELAYSPEC_OUTPUT"] = str(directory)
        subprocess.run(
            [
                sys.executable,
                "scripts/check_sd_square_campaign.py",
                "--config",
                str(config_path),
            ],
            env=env,
            check=True,
            capture_output=True,
        )
        gate = json.loads((directory / "completion-gate.json").read_text())
        if gate != json.loads((run / "completion-gate.json").read_text()):
            raise ValueError("SD-square completion gate does not reproduce")
        config = json.loads(config_path.read_text())
    return config, gate, read_rows(run)


def main():
    parser = argparse.ArgumentParser()
    fitting = parser.add_mutually_exclusive_group(required=True)
    fitting.add_argument("--fits", type=Path, nargs=2)
    fitting.add_argument("--epoch-run", type=Path)
    fitting.add_argument("--quality", action="store_true")
    fitting.add_argument("--quality-full", action="store_true")
    parser.add_argument("--runs", type=Path, nargs=2, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shards = [audit_shard(args, run, i) for i, run in enumerate(args.runs)]
    config, gate, _ = shards[0]
    other_config, other_gate, _ = shards[1]
    if {k: v for k, v in config.items() if k != "request_offset"} != {
        k: v for k, v in other_config.items() if k != "request_offset"
    } or gate["steering_fingerprints"] != other_gate["steering_fingerprints"]:
        raise ValueError("SD-square shards differ in weights or declaration")
    rows = [row for _, _, shard_rows in shards for row in shard_rows]
    quality_inputs, quality = [], None
    if args.quality or args.quality_full:
        scorer = Path("reports/ar-revision-20260905/scorer-provenance.json")
        rows = []
        for run in args.runs:
            analysis_path = run / "analysis.json"
            analysis = json.loads(analysis_path.read_text())
            if (
                analysis["status"] != "complete"
                or analysis["scorer_provenance_sha256"] != digest(scorer)
                or analysis["raw_sha256"]
                != {p.name: digest(p) for p in run.glob("benchmark-rank*.jsonl")}
            ):
                raise ValueError(
                    "SD-square quality analysis differs from raw outputs or pinned scorer"
                )
            rows.extend(load_scored(run))
            quality_inputs.extend([analysis_path, run / "math-scored.jsonl"])
        quality_inputs.extend([scorer, Path("src/relayspec/paired_accuracy.py")])
        paired = {}
        for row in rows:
            paired.setdefault(row["problem_id"], {})[row["method"]] = row["correct"]
        quality = {
            "requests": len(paired),
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
                for name in config["variants"]
            },
            "conservative_paired_accuracy": paired_accuracy_interval(
                [g["sd2_selected"] for g in paired.values()],
                [g["native_ar"] for g in paired.values()],
            ),
            "scope": "Capped exposed development outcomes. No untouched confirmation or final quality-margin claim from this eight-request pilot.",
        }
        if args.quality_full:
            quality["scope"] = (
                "128 capped exposed development outcomes. No untouched confirmation or final "
                "quality-margin claim. Conservative paired intervals supplement descriptive bootstrap intervals."
            )
    summaries = {
        ref: summarize(rows, reference=ref)
        for ref in ("native_ar", "sd2_independent", "sd2_zero_guidance")
        if ref in config["variants"]
    }
    acceptance = {}
    if summaries["native_ar"]["requests"] != config["total_development_requests"]:
        raise ValueError("SD-square shards do not cover the declared disjoint requests")
    for name in config["variants"]:
        selected = [r for r in rows if r["method"] == name]
        proposed = sum(sum(r["proposal_lengths"]) for r in selected)
        accepted = sum(sum(r["accepted_draft_lengths"]) for r in selected)
        acceptance[name] = {
            "proposed": proposed,
            "accepted": accepted,
            "fraction": accepted / proposed if proposed else None,
        }
    result = {
        "status": "complete",
        "input_sha256": {
            str(p): digest(p)
            for p in [
                args.ledger,
                *quality_inputs,
                *[
                    p
                    for run in args.runs
                    for p in [
                        run / "source-commit.txt",
                        run / "campaign-config.json",
                        run / "completion-gate.json",
                        *run.glob("benchmark-rank*.jsonl"),
                        *run.glob("campaign-rank*.json"),
                    ]
                ],
            ]
        },
        "fitting_input_sha256": config["fitting_input_sha256"],
        "variants": config["variants"],
        "inference_precision": gate["inference_precision"],
        "steering_fingerprints": gate["steering_fingerprints"],
        "comparisons": summaries,
        "acceptance": acceptance,
        "ar_exact_counts": {
            m: sum(g["ar_exact_counts"][m] for _, g, _ in shards)
            for m in config["variants"]
        },
        "original_exact_ar_gates": config["original_exact_ar_gates"],
        "scope": config["scope"]
        + " Paired request bootstrap intervals are descriptive and do not account for rate selection or fitting-seed variability.",
    }
    if quality is not None:
        result["capped_quality"] = quality
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summaries["native_ar"]))


if __name__ == "__main__":
    main()
