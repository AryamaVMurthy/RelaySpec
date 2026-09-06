"""Run one shared-reference mapper campaign and verify duplicate-map isolation."""

import argparse
import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--equal-methods", nargs=2)
    args = parser.parse_args()
    python = os.environ["RELAYSPEC_PYTHON"]
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output / "config.yaml")
    config = yaml.safe_load(args.config.read_text())
    family = config["proposer"]["family"]
    if family not in {"dflash", "eagle3"}:
        raise ValueError("mapper campaign requires a supported proposer family")
    configs = [(args.config, output)]
    if config["benchmark"].get("isolate_methods", False):
        configs = []
        for method in config["benchmark"]["methods"]:
            child = copy.deepcopy(config)
            child["benchmark"]["methods"] = [method]
            child["benchmark"].pop("isolate_methods", None)
            if family == "dflash":
                child["benchmark"]["unload_source_trunk"] = method.startswith("relay_")
            probe = child.get("relay_probe", {})
            for key in ("variants", "drafter_updates"):
                if key in probe:
                    probe[key] = {k: v for k, v in probe[key].items() if k == method}
            child_output = output / "isolated" / method
            child_output.mkdir(parents=True, exist_ok=True)
            child_path = child_output / "config.yaml"
            child_path.write_text(yaml.safe_dump(child, sort_keys=False))
            configs.append((child_path, child_output))
    for child_path, child_output in configs:
        subprocess.run(
            [python, "-m", "torch.distributed.run", "--standalone",
             "--nproc_per_node=4",
             "scripts/benchmark_relay.py" if family == "dflash"
             else "scripts/benchmark_eagle3.py", "--config", str(child_path)],
            env={**os.environ, "RELAYSPEC_OUTPUT": str(child_output)}, check=True,
        )
    if config["benchmark"].get("isolate_methods", False):
        for rank in range(4):
            with (output / f"benchmark-rank{rank}.jsonl").open("w") as merged:
                for _, child_output in configs:
                    merged.write((child_output / f"benchmark-rank{rank}.jsonl").read_text())
    subprocess.run(
        [
            python,
            "scripts/check_run_complete.py",
            "--config",
            str(args.config),
            "--output",
            str(output),
        ],
        check=True,
    )
    equality = None
    if args.equal_methods:
        rows = [
            json.loads(line)
            for p in output.glob("benchmark-rank*.jsonl")
            for line in p.read_text().splitlines()
        ]
        maps = [
            {(r["problem_id"], r["repetition"]): r for r in rows if r["method"] == name}
            for name in args.equal_methods
        ]
        if not maps[0] or maps[0].keys() != maps[1].keys():
            raise ValueError("duplicate mapper methods lack the same requests")
        checked_fields = ["output_hash", "output_tokens", "mapper_checkpoint_sha256"]
        checked_fields += (
            ["accepted_draft_lengths", "proposal_lengths"]
            if family == "dflash"
            else ["acceptance_lengths", "target_calls", "draft_calls"]
        )
        for key, first in maps[0].items():
            second = maps[1][key]
            for field in checked_fields:
                if first[field] != second[field]:
                    raise ValueError(
                        f"duplicate-map state isolation failed: {key} {field}"
                    )
        equality = {
            "methods": args.equal_methods,
            "requests": len(maps[0]),
            "status": "pass",
            "checked_fields": checked_fields,
        }
    config = yaml.safe_load(args.config.read_text())
    (output / "campaign-gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "duplicate_map_equivalence": equality,
                "methods": config["benchmark"]["methods"],
                "scope": "Paired artifact completeness and optional identical-checkpoint isolation; not a task-quality or optimality claim.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
