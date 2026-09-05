from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from relayspec.code_eval import extract_code, run_python_tests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=5)
    return parser.parse_args()


def load_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("benchmark-rank*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            rows.extend(
                row
                for line in stream
                if line.strip()
                for row in [json.loads(line)]
                if row["benchmark"] == "mbpp"
            )
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.output_dir)
    if not rows:
        raise ValueError("benchmark output contains no MBPP rows")

    def score(row: dict[str, Any]) -> dict[str, Any]:
        reference = row["reference_answer"]
        result = run_python_tests(
            extract_code(str(row["completion"])),
            list(reference["tests"]),
            timeout_seconds=args.timeout_seconds,
        )
        return {
            "problem_id": row["problem_id"],
            "method": row["method"],
            **result,
        }

    result_path = args.output_dir / "mbpp-base-results.jsonl"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(score, rows))
    with result_path.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
    methods = sorted({str(result["method"]) for result in results})
    summary = {
        "benchmark": "mbpp",
        "scorer": "resource_limited_base_tests",
        "methods": {
            method: {
                "samples": sum(result["method"] == method for result in results),
                "passed": sum(
                    result["method"] == method and result["passed"]
                    for result in results
                ),
            }
            for method in methods
        },
    }
    for method in methods:
        values = summary["methods"][method]
        values["pass_at_1"] = values["passed"] / values["samples"]
    (args.output_dir / "mbpp-base-summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
