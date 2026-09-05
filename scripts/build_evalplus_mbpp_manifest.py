from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def mbpp_evalplus_records(
    problems: Mapping[str, Mapping[str, Any]],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    if limit > len(problems):
        raise ValueError(
            f"requested {limit} EvalPlus MBPP tasks, but only {len(problems)} exist"
        )
    task_ids = sorted(problems)
    random.Random(seed).shuffle(task_ids)
    records: list[dict[str, Any]] = []
    for task_id in task_ids[:limit]:
        problem = problems[task_id]
        assertions = [
            line.strip()
            for line in str(problem["assertion"]).splitlines()
            if line.strip().startswith("assert ")
        ]
        if not assertions:
            raise ValueError(f"{task_id} has no published base assertions")
        prompt = (
            str(problem["prompt"]).rstrip()
            + "\n\nYour code must satisfy all base tests:\n```python\n"
            + "\n".join(assertions)
            + "\n```"
        )
        entry_point = str(problem["entry_point"])
        if entry_point not in prompt:
            raise ValueError(f"{task_id} prompt omits entry point {entry_point!r}")
        records.append(
            {
                "benchmark": "mbpp",
                "problem_id": task_id,
                "prompt": prompt,
                "answer": {
                    "entry_point": entry_point,
                    "tests": assertions,
                },
            }
        )
    return records


def exclude_task_ids(
    records: list[dict[str, Any]], task_ids: set[str]
) -> list[dict[str, Any]]:
    return [record for record in records if str(record["problem_id"]) not in task_ids]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evalplus-source", type=Path, required=True)
    parser.add_argument("--evalplus-revision", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        help="Exclude MBPP problem IDs already present in this manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.evalplus_source.resolve()))
    from evalplus.data import get_mbpp_plus

    payload = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    retained = [
        record for record in payload["records"] if record["benchmark"] != "mbpp"
    ]
    replacement = mbpp_evalplus_records(
        get_mbpp_plus(),
        limit=args.limit,
        seed=args.seed,
    )
    if args.exclude_manifest is not None:
        excluded_payload = json.loads(args.exclude_manifest.read_text(encoding="utf-8"))
        excluded_ids = {
            str(record["problem_id"])
            for record in excluded_payload["records"]
            if record["benchmark"] == "mbpp"
        }
        replacement = exclude_task_ids(replacement, excluded_ids)
    payload["records"] = retained + replacement
    payload["seed"] = args.seed
    payload["dataset_revisions"].pop("mbpp", None)
    payload["dataset_revisions"]["evalplus_mbpp"] = args.evalplus_revision
    serialized = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    args.output.with_suffix(".sha256").write_text(
        f"{digest}  {args.output}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(payload["records"]),
                "mbpp_records": len(replacement),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
