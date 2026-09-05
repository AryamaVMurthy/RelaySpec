#!/usr/bin/env python3
"""Build the per-task breadth matrix against plain autoregressive decoding.

Every ratio here uses the identical statistic as ``MAIN_PAIRED_AR.json``:
``paired_bootstrap_speedup`` over the per-request ``request_seconds`` of a
single benchmark run in which plain autoregressive decoding, optimized
source reuse, and RelaySpec were all measured together on the same prompts.
Because plain AR is measured inside the same run, the ratio is paired
request by request rather than compared across executions.

MT-Bench is clustered by conversation rather than by turn, matching the
bootstrap unit the rest of the paper uses for that task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from relayspec.benchmarking import paired_bootstrap_speedup

REPLICATES = 10_000
SEED = 1729

# Method naming differs between the DFlash and EAGLE-3 benchmark scripts.
REFERENCE = {
    "dflash": ("optimized_source_reuse", "relay_p"),
    "eagle3": ("source_reuse_eagle3", "relay_eagle3"),
}


def _load_rows(result_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rank_files = sorted(result_dir.glob("benchmark-rank*.jsonl"))
    if not rank_files:
        raise FileNotFoundError(f"no benchmark-rank*.jsonl under {result_dir}")
    for path in rank_files:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _cluster_key(row: dict[str, Any]) -> str:
    """The bootstrap unit: an MT-Bench conversation, or one problem elsewhere.

    MT-Bench ``problem_id`` is ``mtbench/<conversation>/turn<n>``. The two
    turns of one conversation share a prefix and are strongly dependent, so
    resampling them independently would understate the interval. Stripping
    the trailing turn segment gives 80 conversations rather than 160 turns,
    which is the bootstrap unit the rest of the paper already uses for this
    task.
    """
    problem = str(row["problem_id"])
    if str(row["benchmark"]) == "mtbench":
        return problem.rsplit("/", 1)[0]
    return problem


def _speedup(
    rows: list[dict[str, Any]], reference: str, candidate: str
) -> dict[str, float]:
    """Paired bootstrap ratio, summing every turn inside one cluster first."""
    totals: dict[tuple[str, int, str], float] = {}
    for row in rows:
        method = str(row["method"])
        if method not in {reference, candidate}:
            continue
        key = (_cluster_key(row), int(row["repetition"]), method)
        totals[key] = totals.get(key, 0.0) + float(row["request_seconds"])
    collapsed = [
        {
            "method": method,
            "problem_id": problem,
            "repetition": repetition,
            "request_seconds": seconds,
        }
        for (problem, repetition, method), seconds in totals.items()
    ]
    result = paired_bootstrap_speedup(
        collapsed,
        reference_method=reference,
        candidate_method=candidate,
        replicates=REPLICATES,
        seed=SEED,
    )
    return {key: float(result[key]) for key in ("estimate", "lower", "upper")}


def _mean_acceptance(rows: list[dict[str, Any]], method: str) -> float:
    """Micro-average accepted tokens per cycle over every proposal cycle."""
    accepted = 0
    cycles = 0
    for row in rows:
        if str(row["method"]) != method:
            continue
        lengths = row.get("acceptance_lengths")
        if not lengths:
            continue
        accepted += sum(int(value) for value in lengths)
        cycles += len(lengths)
    if cycles == 0:
        raise ValueError(f"no acceptance cycles recorded for {method}")
    return accepted / cycles


def build_cell(family: str, target: str, result_dir: Path) -> dict[str, Any]:
    reference, candidate = REFERENCE[family]
    rows = _load_rows(result_dir)
    methods = sorted({str(row["method"]) for row in rows})
    for required in ("native_ar", reference, candidate):
        if required not in methods:
            raise ValueError(
                f"{result_dir}: missing method {required}, found {methods}"
            )
    benchmarks = sorted({str(row["benchmark"]) for row in rows})
    tasks: dict[str, Any] = {}
    for benchmark in benchmarks:
        subset = [row for row in rows if str(row["benchmark"]) == benchmark]
        clusters = len({_cluster_key(row) for row in subset})
        requests = len({str(row["problem_id"]) for row in subset})
        tasks[benchmark] = {
            "requests": requests,
            "bootstrap_clusters": clusters,
            "relay_vs_ar": _speedup(subset, "native_ar", candidate),
            "source_reuse_vs_ar": _speedup(subset, "native_ar", reference),
            "relay_vs_source_reuse": _speedup(subset, reference, candidate),
            "accept_source_reuse": _mean_acceptance(subset, reference),
            "accept_relay": _mean_acceptance(subset, candidate),
        }
    return {"family": family, "target": target, "tasks": tasks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        nargs=3,
        action="append",
        metavar=("FAMILY", "TARGET", "RESULT_DIR"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cells = [
        build_cell(family, target, Path(result_dir))
        for family, target, result_dir in args.cell
    ]
    payload = {
        "_provenance": (
            "Every ratio is a paired bootstrap over request clusters "
            "(paired_bootstrap_speedup, ratio of summed paired request time, "
            f"{REPLICATES:,} replicates, seed {SEED}) computed from the "
            "per-rank benchmark-rank*.jsonl rows of a single benchmark run "
            "per proposer and target, in which plain autoregressive "
            "decoding, optimized source reuse and RelaySpec were all "
            "measured together on the same prompts. MT-Bench is clustered "
            "by conversation. Accepted tokens per cycle are micro-averages "
            "over every proposal cycle in the run."
        ),
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
