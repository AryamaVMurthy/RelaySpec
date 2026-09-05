"""Require the exact planned method/request set before a dependent job runs."""

import argparse
import json
import math
import random
from pathlib import Path

import yaml

from relayspec.benchmarking import benchmark_turns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    manifest = json.loads(Path(config["benchmark"]["manifest_path"]).read_text())
    selected = [
        r
        for r in manifest["records"]
        if r["benchmark"] in config["benchmark"]["benchmarks"]
    ]
    random.Random(config["seed"]).shuffle(selected)
    selected = selected[: config["benchmark"]["max_prompts"]]
    expected = set()
    for record in selected:
        turns = benchmark_turns(record)
        for turn in range(len(turns)):
            pid = record["problem_id"] + (f"/turn{turn}" if len(turns) > 1 else "")
            for repeat in range(config.get("relay_probe", {}).get("repetitions", 1)):
                for method in config["benchmark"]["methods"]:
                    expected.add((pid, repeat, method))
    rows = [
        json.loads(line)
        for path in args.output.glob("benchmark-rank*.jsonl")
        for line in path.open()
        if line.strip()
    ]
    actual = {(r["problem_id"], r["repetition"], r["method"]) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError(
            f"Incomplete run: expected {len(expected)} unique records, got {len(rows)}"
        )
    if any(
        not math.isfinite(r["request_seconds"])
        or r["request_seconds"] <= 0
        or r["output_tokens"] <= 0
        for r in rows
    ):
        raise ValueError("Invalid time or output length")
    report = {
        "status": "pass",
        "records": len(rows),
        "requests": len({(r["problem_id"], r["repetition"]) for r in rows}),
        "scope": "Artifact completeness and finite measurements. Output equivalence is checked separately.",
    }
    (args.output / "completion-gate.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
