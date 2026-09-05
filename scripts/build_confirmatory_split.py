from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def normalized_prompt_hash(record: dict[str, Any]) -> str:
    normalized = " ".join(str(record["prompt"]).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_confirmatory_split(
    development_manifest: dict[str, Any],
    full_manifest: dict[str, Any],
    *,
    seed: int,
    development_count: int,
) -> dict[str, Any]:
    development = [
        row for row in development_manifest["records"] if row["benchmark"] == "math500"
    ]
    random.Random(seed).shuffle(development)
    selected_hashes = {
        normalized_prompt_hash(row) for row in development[:development_count]
    }
    full_math = [
        row for row in full_manifest["records"] if row["benchmark"] == "math500"
    ]
    excluded = [
        row for row in full_math if normalized_prompt_hash(row) in selected_hashes
    ]
    if len(excluded) != development_count:
        raise ValueError(
            "development/full overlap does not match the registered selection "
            f"count: {len(excluded)} != {development_count}"
        )
    kept = [
        row for row in full_math if normalized_prompt_hash(row) not in selected_hashes
    ]
    if len(kept) + len(excluded) != len(full_math):
        raise AssertionError("confirmatory split does not partition MATH-500")
    return {
        "split": "math500_confirmatory_complement",
        "selection_seed": seed,
        "development_count": development_count,
        "excluded_problem_ids": sorted(str(row["problem_id"]) for row in excluded),
        "records": kept,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--development-count", type=int, default=32)
    args = parser.parse_args()
    result = build_confirmatory_split(
        json.loads(args.development_manifest.read_text(encoding="utf-8")),
        json.loads(args.full_manifest.read_text(encoding="utf-8")),
        seed=args.seed,
        development_count=args.development_count,
    )
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(result["records"]),
                "excluded": len(result["excluded_problem_ids"]),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
