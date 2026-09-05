from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from relayspec.code_eval import extract_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def evalplus_task_id(benchmark: str, problem_id: str) -> str:
    if benchmark == "humaneval":
        return problem_id
    if benchmark == "mbpp":
        suffix = problem_id.split("/", 1)[-1]
        return f"Mbpp/{suffix}"
    raise ValueError(f"unsupported EvalPlus benchmark: {benchmark}")


def load_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("benchmark-rank*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def main() -> None:
    args = parse_args()
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in load_rows(args.output_dir):
        benchmark = str(row["benchmark"])
        if benchmark not in {"humaneval", "mbpp"}:
            continue
        method = str(row["method"])
        grouped[(benchmark, method)].append(
            {
                "task_id": evalplus_task_id(benchmark, str(row["problem_id"])),
                "solution": extract_code(str(row["completion"])),
            }
        )
    export_dir = args.output_dir / "evalplus"
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"files": []}
    for (benchmark, method), samples in sorted(grouped.items()):
        samples.sort(key=lambda row: row["task_id"])
        path = export_dir / f"{benchmark}-{method}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for sample in samples:
                stream.write(json.dumps(sample, sort_keys=True) + "\n")
        manifest["files"].append(
            {
                "benchmark": benchmark,
                "method": method,
                "path": str(path),
                "samples": len(samples),
            }
        )
    (export_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
