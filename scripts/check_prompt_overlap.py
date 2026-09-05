from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def normalize_prompt(text: str) -> str:
    """Normalize layout only; mathematical and programming operators are semantic."""
    return " ".join(text.split())


def prompt_hash(text: str) -> str:
    return hashlib.sha256(normalize_prompt(text).encode("utf-8")).hexdigest()


def _fit_text(record: dict[str, Any]) -> str:
    for key in ("problem", "prompt", "question"):
        if key in record:
            return str(record[key])
    raise ValueError("fit record has no problem, prompt, or question field")


def _evaluation_texts(record: dict[str, Any]) -> list[str]:
    if "prompt" in record:
        return [str(record["prompt"])]
    turns = record.get("turns")
    if isinstance(turns, list) and turns:
        return [str(turn) for turn in turns]
    raise ValueError("evaluation record has neither prompt nor non-empty turns")


def audit_prompt_overlap(
    fit_manifest: dict[str, Any], evaluation_manifest: dict[str, Any]
) -> dict[str, Any]:
    fit_records = list(fit_manifest["records"])
    evaluation_records = list(evaluation_manifest["records"])
    fit_by_hash: dict[str, list[int]] = {}
    for index, record in enumerate(fit_records):
        fit_by_hash.setdefault(prompt_hash(_fit_text(record)), []).append(index)

    overlaps: list[dict[str, Any]] = []
    by_benchmark: dict[str, int] = {}
    for record in evaluation_records:
        matching_indices = sorted(
            {
                fit_index
                for text in _evaluation_texts(record)
                for fit_index in fit_by_hash.get(prompt_hash(text), [])
            }
        )
        if not matching_indices:
            continue
        benchmark = str(record["benchmark"])
        by_benchmark[benchmark] = by_benchmark.get(benchmark, 0) + 1
        overlaps.append(
            {
                "benchmark": benchmark,
                "problem_id": str(record["problem_id"]),
                "fit_record_indices": matching_indices,
            }
        )
    return {
        "normalization": "unicode-preserving whitespace collapse",
        "fit_records": len(fit_records),
        "evaluation_records": len(evaluation_records),
        "overlap_count": len(overlaps),
        "overlap_count_by_benchmark": dict(sorted(by_benchmark.items())),
        "overlaps": overlaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = audit_prompt_overlap(
        json.loads(args.fit_manifest.read_text(encoding="utf-8")),
        json.loads(args.evaluation_manifest.read_text(encoding="utf-8")),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if result["overlap_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
