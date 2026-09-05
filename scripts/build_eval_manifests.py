from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

DATASETS = {
    "math500": (
        "HuggingFaceH4/MATH-500",
        None,
        "test",
        "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    ),
    "gsm8k": (
        "openai/gsm8k",
        "main",
        "test",
        "740312add88f781978c0658806c59bc2815b9866",
    ),
    "humaneval": (
        "openai/openai_humaneval",
        None,
        "test",
        "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
    ),
    "mbpp": (
        "google-research-datasets/mbpp",
        "sanitized",
        "test",
        "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    ),
    "mtbench": (
        "HuggingFaceH4/mt_bench_prompts",
        None,
        "train",
        "e3a795c5e9a82ee40611c416b8a7786c73198991",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stable_subset(
    rows: list[dict[str, Any]], *, limit: int, seed: int
) -> list[dict[str, Any]]:
    selected = list(rows)
    random.Random(seed).shuffle(selected)
    return selected[:limit]


def gsm8k_answer(value: str) -> str:
    if "####" not in value:
        raise ValueError("GSM8K answer is missing the official delimiter")
    return value.rsplit("####", 1)[1].strip().replace(",", "")


def mbpp_prompt(description: str, tests: list[str]) -> str:
    """Build an executable-specification prompt for an MBPP task.

    The sanitized MBPP descriptions frequently omit the required function
    name.  The published base assertions are therefore part of the task
    prompt, as in standard MBPP evaluation, rather than hidden metadata.
    """
    if not tests:
        raise ValueError("MBPP requires at least one published base test")
    return (
        description.rstrip()
        + "\n\nYour code must satisfy these tests:\n```python\n"
        + "\n".join(tests)
        + "\n```"
    )


def load_rows(name: str, cache_dir: Path) -> list[dict[str, Any]]:
    from datasets import load_dataset

    repo_id, subset, split, revision = DATASETS[name]
    args = (repo_id,) if subset is None else (repo_id, subset)
    dataset = load_dataset(
        *args,
        split=split,
        revision=revision,
        cache_dir=str(cache_dir),
    )
    return [dict(row) for row in dataset]


def build_records(cache_dir: Path, seed: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(load_rows("math500", cache_dir)):
        records.append(
            {
                "benchmark": "math500",
                "problem_id": f"math500/{index}",
                "prompt": row["problem"],
                "answer": row["answer"],
            }
        )
    gsm8k = stable_subset(load_rows("gsm8k", cache_dir), limit=128, seed=seed)
    for row in gsm8k:
        records.append(
            {
                "benchmark": "gsm8k",
                "problem_id": f"gsm8k/{row.get('idx', hashlib.sha256(row['question'].encode()).hexdigest()[:16])}",
                "prompt": row["question"],
                "answer": gsm8k_answer(row["answer"]),
            }
        )
    for row in load_rows("humaneval", cache_dir):
        records.append(
            {
                "benchmark": "humaneval",
                "problem_id": str(row["task_id"]),
                "prompt": (
                    "Write a solution to the following problem and make sure that "
                    f"it passes the tests:\n```python\n{row['prompt']}\n```"
                ),
                "answer": {
                    "entry_point": row["entry_point"],
                    "test": row["test"],
                },
            }
        )
    mbpp = stable_subset(load_rows("mbpp", cache_dir), limit=200, seed=seed)
    for row in mbpp:
        tests = list(row["test_list"])
        records.append(
            {
                "benchmark": "mbpp",
                "problem_id": f"mbpp/{row['task_id']}",
                "prompt": mbpp_prompt(str(row["prompt"]), tests),
                "answer": {
                    "entry_point": row.get("entry_point"),
                    "tests": tests,
                },
            }
        )
    for index, row in enumerate(load_rows("mtbench", cache_dir)):
        records.append(
            {
                "benchmark": "mtbench",
                "problem_id": f"mtbench/{index}",
                "turns": list(row["prompt"]),
                "answer": None,
            }
        )
    return records


def main() -> None:
    args = parse_args()
    records = build_records(args.cache_dir, args.seed)
    payload = {
        "seed": args.seed,
        "dataset_revisions": {name: values[3] for name, values in DATASETS.items()},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(records),
                "sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
