from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml


def load_rows(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("benchmark-rank*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    if not rows:
        raise ValueError(f"no benchmark rank rows in {directory}")
    return rows


def merge_rows(
    first: Sequence[dict[str, Any]],
    remainder: Sequence[dict[str, Any]],
    *,
    expected_problem_ids: set[str],
    methods: Sequence[str],
) -> list[dict[str, Any]]:
    method_names = tuple(methods)
    expected = {
        (problem_id, method, 0)
        for problem_id in expected_problem_ids
        for method in method_names
    }
    observed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in [*first, *remainder]:
        if row.get("benchmark") != "mbpp":
            raise ValueError("merged benchmark rows must all be MBPP")
        key = (
            str(row["problem_id"]),
            str(row["method"]),
            int(row.get("repetition", 0)),
        )
        if key in observed:
            raise ValueError(f"duplicate merged benchmark row: {key}")
        observed[key] = row
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    if missing:
        raise ValueError(f"missing merged task/method rows: {missing[:5]}")
    if extra:
        raise ValueError(f"unexpected merged task/method rows: {extra[:5]}")
    method_order = {method: index for index, method in enumerate(method_names)}
    return sorted(
        observed.values(),
        key=lambda row: (
            str(row["problem_id"]),
            int(row.get("repetition", 0)),
            method_order[str(row["method"])],
        ),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-dir", type=Path, required=True)
    parser.add_argument("--remainder-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"merge output already exists: {args.output_dir}")
    if args.world_size <= 0:
        raise ValueError("world size must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_ids = {
        str(record["problem_id"])
        for record in manifest["records"]
        if record["benchmark"] == "mbpp"
    }
    methods = tuple(str(method) for method in config["benchmark"]["methods"])
    rows = merge_rows(
        load_rows(args.first_dir),
        load_rows(args.remainder_dir),
        expected_problem_ids=expected_ids,
        methods=methods,
    )
    args.output_dir.mkdir(parents=True)
    streams = [
        (args.output_dir / f"benchmark-rank{rank}.jsonl").open("x", encoding="utf-8")
        for rank in range(args.world_size)
    ]
    try:
        by_problem: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_problem.setdefault(str(row["problem_id"]), []).append(row)
        for index, problem_id in enumerate(sorted(by_problem)):
            stream = streams[index % args.world_size]
            for row in by_problem[problem_id]:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    finally:
        for stream in streams:
            stream.close()
    shutil.copy2(args.config, args.output_dir / "config.yaml")
    shutil.copy2(args.manifest, args.output_dir / "eval-manifest.json")
    (args.output_dir / "eval-manifest.sha256").write_text(
        f"{file_sha256(args.manifest)}  eval-manifest.json\n",
        encoding="utf-8",
    )
    input_files = sorted(
        [*args.first_dir.glob("benchmark-rank*.jsonl")]
        + [*args.remainder_dir.glob("benchmark-rank*.jsonl")]
    )
    provenance = {
        "first_dir": str(args.first_dir.resolve()),
        "remainder_dir": str(args.remainder_dir.resolve()),
        "methods": methods,
        "problem_ids": len(expected_ids),
        "rows": len(rows),
        "input_sha256": {
            str(path.resolve()): file_sha256(path) for path in input_files
        },
    }
    (args.output_dir / "merge-provenance.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    for label, directory in (
        ("first", args.first_dir),
        ("remainder", args.remainder_dir),
    ):
        for filename in ("allocation.jsonl", "gpu.csv", "source-sha256.txt"):
            source = directory / filename
            if source.exists():
                shutil.copy2(source, args.output_dir / f"{label}-{filename}")
    print(json.dumps(provenance, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
