"""Audit collected fit gates and expose completed and missing study cells."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(raw_root, matrix_path):
    matrix = json.loads(matrix_path.read_text())
    expected = {
        t["name"]: t
        for batches in matrix["groups"].values()
        for batch in batches
        for t in batch
    }
    expected.update({r["trial"]["name"]: r["trial"] for r in matrix["reuse"]})
    cache_hash = None
    completed = {}
    search_root = raw_root / "reports/mapper-scaling-20260905"
    for gate_path in sorted(search_root.glob("*/run-*/batch-gate.json")):
        gate = json.loads(gate_path.read_text())
        if gate.get("status") != "pass" or gate.get("mode") != "fit":
            continue
        for folder in sorted((gate_path.parent / "fitting").glob("*")):
            if folder.name not in expected:
                continue
            fit_path = folder / "fit-complete.json"
            fit = json.loads(fit_path.read_text())
            t = fit["trial"]
            if fit["status"] != "pass" or t != expected[folder.name]:
                raise ValueError(f"fit differs from declared cell: {folder}")
            if folder.name in completed:
                raise ValueError(
                    f"duplicate result requires explicit resolution: {folder.name}"
                )
            cache_hash = cache_hash or gate["feature_cache_index_sha256"]
            if (
                fit["feature_cache_index_sha256"] != cache_hash
                or gate["feature_cache_index_sha256"] != cache_hash
            ):
                raise ValueError("fitting results use different frozen features")
            val_path = folder / "validation.jsonl"
            points = [json.loads(line) for line in val_path.read_text().splitlines()]
            if points[-1]["step"] != t["steps"]:
                raise ValueError("validation endpoint missing")
            if not set(t["checkpoint_steps"]).issubset({p["step"] for p in points}):
                raise ValueError("declared validation checkpoints missing")
            hashes = {
                Path(p).name: sha
                for p, sha in gate["checkpoint_sha256"].items()
                if Path(p).parent.name == folder.name
            }
            if {f"step-{s:06d}.pt" for s in t["checkpoint_steps"]} != set(hashes):
                raise ValueError("checkpoint completion hashes missing")
            endpoint = points[-1]["groups"]
            completed[folder.name] = {
                "trial": t,
                "parameters": fit["parameters"],
                "train_objective": endpoint["train"]["objective"],
                "validation_objective": endpoint["validation"]["objective"],
                "best_validation_step": min(
                    points, key=lambda p: p["groups"]["validation"]["objective"]
                )["step"],
                "validation_trajectory": points,
                "timing": {
                    k: fit[k]
                    for k in [
                        "loop_seconds",
                        "setup_seconds",
                        "input_io_seconds",
                        "validation_seconds",
                    ]
                },
                "peak_gpu_bytes": fit.get("peak_gpu_bytes"),
                "raw_directory": str(folder.relative_to(raw_root)),
                "fit_sha256": digest(fit_path),
                "validation_sha256": digest(val_path),
                "batch_gate_sha256": digest(gate_path),
                "checkpoint_sha256": hashes,
            }
    return {
        "status": "complete" if len(completed) == len(expected) else "incomplete",
        "expected_cells": len(expected),
        "completed_cells": len(completed),
        "missing_cells": sorted(set(expected) - set(completed)),
        "matrix_sha256": digest(matrix_path),
        "feature_cache_index_sha256": cache_hash,
        "results": completed,
        "scope": "Same-cache Numina fitting diagnostics at N512/2048. A fit gate establishes checkpoint completion, not decoding speed or answer quality. Minimum validation points use development data. Loop time includes diagnostics and checkpoint writes; I/O timing measures the host-side data preparation region.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("configs/submission/scaling/matrix-focused-v1/matrix.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.raw_root, args.matrix)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ["status", "expected_cells", "completed_cells"]}
        )
    )


if __name__ == "__main__":
    main()
