"""Run pinned EvalPlus tests on exactly the supplied task IDs, with separate caches."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path


def select_tasks(problems, task_ids):
    if len(task_ids) != len(set(task_ids)) or not task_ids:
        raise ValueError("pass@1 requires one sample per task and a nonempty set")
    missing = set(task_ids) - set(problems)
    if missing:
        raise ValueError(f"tasks unavailable in official EvalPlus: {sorted(missing)}")
    return {key: problems[key] for key in sorted(task_ids)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["humaneval", "mbpp"])
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.samples.read_text().splitlines() if line]
    task_ids = [row["task_id"] for row in rows]
    official = importlib.import_module("evalplus.evaluate")
    loader_name = (
        "get_human_eval_plus" if args.dataset == "humaneval" else "get_mbpp_plus"
    )
    hash_name = loader_name + "_hash"
    loader = getattr(official, loader_name)
    hasher = getattr(official, hash_name)
    problems = select_tasks(loader(), task_ids)
    full_hash = hasher()
    subset_hash = hashlib.sha256(
        json.dumps([full_hash, sorted(task_ids)]).encode()
    ).hexdigest()
    # Only data selection changes. Official base/plus inputs, canonical oracles,
    # timeouts, sandboxed checks and aggregation remain untouched. Never put a
    # subset ground-truth pickle under the full dataset's cache key.
    setattr(official, loader_name, lambda **kwargs: problems)
    setattr(official, hash_name, lambda **kwargs: subset_hash)
    provenance = {
        "dataset": args.dataset,
        "task_ids": sorted(task_ids),
        "full_dataset_hash": full_hash,
        "subset_hash": subset_hash,
        "samples_sha256": hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "upstream_evaluator_sha256": hashlib.sha256(
            Path(official.__file__).read_bytes()
        ).hexdigest(),
        "scope": "Official complete base and plus test cases for only the supplied task IDs; no missing-task placeholders or denominator changes.",
    }
    args.output_file.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    official.evaluate(
        args.dataset,
        samples=str(args.samples),
        parallel=args.parallel,
        i_just_wanna_run=True,
        output_file=str(args.output_file),
    )
    result = json.loads(args.output_file.read_text())
    if set(result["eval"]) != set(task_ids):
        raise ValueError("scorer result IDs do not match supplied tasks")


if __name__ == "__main__":
    main()
