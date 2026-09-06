"""Rebuild the SD-square declaration and decision gate from raw fitting/decoding."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from relayspec.ar_paper_evidence import read_rows, summarize


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
        subprocess.run(
            [
                sys.executable,
                "scripts/build_sd_square_campaign.py",
                "--shard-index",
                str(index),
                "--runs",
                *map(str, args.fits),
                "--output",
                str(config_path),
            ],
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
    parser.add_argument("--fits", type=Path, nargs=2, required=True)
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
    summaries = {
        ref: summarize(rows, reference=ref)
        for ref in ("native_ar", "sd2_independent", "sd2_zero_guidance")
    }
    acceptance = {}
    if summaries["native_ar"]["requests"] != 16:
        raise ValueError("SD-square shards do not cover sixteen disjoint requests")
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
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summaries["native_ar"]))


if __name__ == "__main__":
    main()
