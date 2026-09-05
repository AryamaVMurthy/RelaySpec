from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MEMORY_FIELDS = (
    "allocated_memory_before_bytes",
    "reserved_memory_before_bytes",
    "peak_allocated_memory_bytes",
    "peak_reserved_memory_bytes",
)
FIELD_LABELS = {
    "allocated_memory_before_bytes": "allocated_before",
    "reserved_memory_before_bytes": "reserved_before",
    "peak_allocated_memory_bytes": "peak_allocated",
    "peak_reserved_memory_bytes": "peak_reserved",
}


def load_rows(directory: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for path in sorted(directory.glob("benchmark-rank*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows:
        raise ValueError(f"no benchmark rows in {directory}")
    return rows


def summarize_memory_pair(
    source_rows: list[dict[str, Any]],
    relay_rows: list[dict[str, Any]],
    *,
    expected_requests: int,
) -> dict[str, Any]:
    if len(source_rows) != expected_requests or len(relay_rows) != expected_requests:
        raise ValueError(
            f"isolated memory runs must each contain {expected_requests} requests"
        )
    source_methods = {str(row["method"]) for row in source_rows}
    relay_methods = {str(row["method"]) for row in relay_rows}
    if len(source_methods) != 1 or len(relay_methods) != 1:
        raise ValueError("each isolated memory run must contain exactly one method")
    source_ids = {
        (str(row["problem_id"]), int(row.get("repetition", 0))) for row in source_rows
    }
    relay_ids = {
        (str(row["problem_id"]), int(row.get("repetition", 0))) for row in relay_rows
    }
    if source_ids != relay_ids or len(source_ids) != expected_requests:
        raise ValueError("isolated memory runs do not cover the same requests")

    result: dict[str, Any] = {
        "source_method": next(iter(source_methods)),
        "relay_method": next(iter(relay_methods)),
        "paired_problem_ids": len(source_ids),
        "source_tokens_per_second": sum(
            int(row["output_tokens"]) for row in source_rows
        )
        / sum(float(row["request_seconds"]) for row in source_rows),
        "relay_tokens_per_second": sum(int(row["output_tokens"]) for row in relay_rows)
        / sum(float(row["request_seconds"]) for row in relay_rows),
    }
    for field in MEMORY_FIELDS:
        source_value = max(int(row[field]) for row in source_rows)
        relay_value = max(int(row[field]) for row in relay_rows)
        label = FIELD_LABELS[field]
        saved = source_value - relay_value
        result[f"source_{label}_bytes"] = source_value
        result[f"relay_{label}_bytes"] = relay_value
        result[f"{label}_saved_bytes"] = saved
        result[f"{label}_saved_gib"] = saved / (1 << 30)
        result[f"{label}_saved_fraction"] = saved / source_value
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Isolated source-versus-relay memory residency",
        "",
        (
            f"Separate processes evaluate `{result['source_method']}` and "
            f"`{result['relay_method']}` on the same "
            f"{result['paired_problem_ids']} requests."
        ),
        "",
        "| Metric | Source | Relay | Saved | Saved fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for field in MEMORY_FIELDS:
        label = FIELD_LABELS[field]
        lines.append(
            f"| {label.replace('_', ' ')} | "
            f"{result[f'source_{label}_bytes'] / (1 << 30):.3f} GiB | "
            f"{result[f'relay_{label}_bytes'] / (1 << 30):.3f} GiB | "
            f"{result[f'{label}_saved_gib']:.3f} GiB | "
            f"{result[f'{label}_saved_fraction']:.2%} |"
        )
    lines.extend(
        [
            "",
            (
                "The isolated micro-throughputs are contextual checks, not a paired "
                f"speed estimate: source {result['source_tokens_per_second']:.3f} "
                f"tok/s, relay {result['relay_tokens_per_second']:.3f} tok/s."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--relay-dir", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_memory_pair(
        load_rows(args.source_dir),
        load_rows(args.relay_dir),
        expected_requests=args.expected_requests,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
