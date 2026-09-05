#!/usr/bin/env python3
"""Measure set-token similarity between relay fitting and evaluation prompts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
DEFAULT_THRESHOLDS = (0.80, 0.90, 0.95)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def unique_tokens(text: str) -> set[str]:
    """Return lowercase Unicode words, numbers, and individual punctuation marks."""
    return set(TOKEN_PATTERN.findall(text.lower()))


def _fit_text(record: dict[str, Any]) -> str:
    for key in ("problem", "prompt", "question"):
        if key in record:
            return str(record[key])
    raise ValueError("fit record has no problem, prompt, or question field")


def _evaluation_text(record: dict[str, Any]) -> str:
    if "prompt" in record:
        return str(record["prompt"])
    turns = record.get("turns")
    if isinstance(turns, list) and turns:
        return "\n".join(str(turn) for turn in turns)
    raise ValueError("evaluation record has neither prompt nor non-empty turns")


def _token_masks(texts: list[str], vocabulary: dict[str, int]) -> list[int]:
    masks: list[int] = []
    for text in texts:
        mask = 0
        for token in unique_tokens(text):
            token_id = vocabulary.setdefault(token, len(vocabulary))
            mask |= 1 << token_id
        masks.append(mask)
    return masks


def _threshold_key(value: float) -> str:
    return f"{value:.2f}"


def audit_prompt_similarity(
    fit_manifest: dict[str, Any],
    evaluation_manifest: dict[str, Any],
    *,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    top_k: int = 5,
) -> dict[str, Any]:
    """Keep the largest unique-token Jaccard score for each evaluation record."""
    if not thresholds or any(value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("thresholds must contain values in [0, 1]")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    fit_records = list(fit_manifest["records"])
    evaluation_records = list(evaluation_manifest["records"])
    vocabulary: dict[str, int] = {}
    fit_masks = _token_masks([_fit_text(record) for record in fit_records], vocabulary)
    fit_sizes = [mask.bit_count() for mask in fit_masks]
    evaluation_masks = _token_masks(
        [_evaluation_text(record) for record in evaluation_records], vocabulary
    )

    rows_by_benchmark: dict[str, list[dict[str, Any]]] = {}
    for record, evaluation_mask in zip(
        evaluation_records, evaluation_masks, strict=True
    ):
        evaluation_size = evaluation_mask.bit_count()
        best_score = -1.0
        best_fit_index = -1
        for fit_index, (fit_mask, fit_size) in enumerate(
            zip(fit_masks, fit_sizes, strict=True)
        ):
            intersection = (evaluation_mask & fit_mask).bit_count()
            union = evaluation_size + fit_size - intersection
            score = intersection / union if union else 1.0
            if score > best_score:
                best_score = score
                best_fit_index = fit_index
        benchmark = str(record["benchmark"])
        rows_by_benchmark.setdefault(benchmark, []).append(
            {
                "problem_id": str(record["problem_id"]),
                "maximum_similarity": best_score,
                "fit_record_index": best_fit_index,
            }
        )

    by_benchmark: dict[str, dict[str, Any]] = {}
    for benchmark, rows in sorted(rows_by_benchmark.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                -float(row["maximum_similarity"]),
                str(row["problem_id"]),
                int(row["fit_record_index"]),
            ),
        )
        scores = [float(row["maximum_similarity"]) for row in rows]
        by_benchmark[benchmark] = {
            "evaluation_records": len(rows),
            "maximum": max(scores),
            "counts": {
                _threshold_key(threshold): sum(score >= threshold for score in scores)
                for threshold in thresholds
            },
            "top_pairs": ordered[:top_k],
        }

    return {
        "method": (
            "lowercase, tokenize Unicode words and individual punctuation, "
            "deduplicate tokens, then compute Jaccard similarity"
        ),
        "multiturn_policy": "join all turns in one evaluation record with a newline",
        "fit_records": len(fit_records),
        "evaluation_records": len(evaluation_records),
        "thresholds": list(thresholds),
        "by_benchmark": by_benchmark,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    result = audit_prompt_similarity(
        read_json(args.fit_manifest), read_json(args.evaluation_manifest)
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Audited {result['evaluation_records']} evaluation records against "
        f"{result['fit_records']} fitting records"
    )


if __name__ == "__main__":
    main()
