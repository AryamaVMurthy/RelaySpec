"""Audit all predeclared small-data 14B fits before endpoint decoding or paper use."""

import argparse
import json
from pathlib import Path

import yaml

from relayspec.cached_fit_evidence import audit_fit_artifacts, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["dflash", "eagle3"], required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = Path("reports/mapper-scaling-20260905")
    config_root = Path("configs/submission/scaling/target14b-small-v1")
    matrix_path = config_root / f"matrix-{args.family}/matrix.json"
    template_path = config_root / f"campaign-{args.family}-pilot.yaml"
    ledger_path = (
        base
        / f"target14b-{'dflash' if args.family == 'dflash' else 'eagle'}-fits/jobs.json"
    )
    cache_path = base / "target14b-cache-results.json"
    pilot_ledger_path = base / "target14b-capacity-pilots/jobs.json"
    matrix, ledger, caches = [
        json.loads(p.read_text()) for p in [matrix_path, ledger_path, cache_path]
    ]
    template = yaml.safe_load(template_path.read_text())
    inputs = {
        str(p): digest(p) for p in [matrix_path, template_path, ledger_path, cache_path]
    }
    if matrix["protocol_sha256"] != digest(config_root / "protocol.json"):
        raise ValueError("14B matrix protocol changed")
    inputs[str(config_root / "protocol.json")] = matrix["protocol_sha256"]
    for name, sha in caches["input_sha256"].items():
        if digest(Path(name)) != sha:
            raise ValueError("cache registry source changed")
        inputs[name] = sha
    cache = next(r for r in caches["records"] if r["family"] == args.family)
    pilot_ledger = json.loads(pilot_ledger_path.read_text())
    pilot_job = next(j for j in pilot_ledger["jobs"] if j["family"] == args.family)
    pilot_path = args.raw_root / pilot_job["local"] / "batch-gate.json"
    pilot_gate = json.loads(pilot_path.read_text())
    if (
        pilot_gate.get("status") != "pass"
        or pilot_gate.get("mode") != "pilot"
        or pilot_gate["feature_cache_index_sha256"] != cache["cache_index_sha256"]
    ):
        raise ValueError("matching exact-cache capacity pilot has not passed")
    inputs[str(pilot_ledger_path)] = digest(pilot_ledger_path)
    inputs[str(pilot_path)] = digest(pilot_path)
    if cache["status"] != "pass" or cache["counts"] != {
        "train": 2048,
        "validation": 1024,
    }:
        raise ValueError("14B cache must have the declared small-data counts")
    declared = matrix["primary_cells"] + matrix["dense_seed_controls"]
    if (
        len(matrix["primary_cells"]) != 14
        or len(matrix["dense_seed_controls"]) != 2
        or len({t["name"] for t in declared}) != 16
        or any(
            t["distinct_examples"] not in {512, 2048} or t["steps"] != 8192
            for t in declared
        )
    ):
        raise ValueError("14B replication requires 14 primary and two dense seed fits")
    expected = {t["name"]: t for t in declared}
    results, batches = {}, []
    for job in ledger["jobs"]:
        folder = args.raw_root / job["local"]
        gate_path = folder / "batch-gate.json"
        gate = json.loads(gate_path.read_text())
        trials_path = Path(job["trials"])
        trials = json.loads(trials_path.read_text())["trials"]
        if (
            gate.get("status") != "pass"
            or gate.get("mode") != "fit"
            or gate["trials"] != [t["name"] for t in trials]
            or gate["trials_sha256"] != digest(trials_path)
            or digest(folder / "trials.json") != digest(trials_path)
            or gate["campaign_config_sha256"] != digest(template_path)
            or gate["feature_cache_index_sha256"] != cache["cache_index_sha256"]
            or gate["pilot_gate_sha256"] != digest(pilot_path)
            or (folder / "source-commit.txt").read_text().strip()
            != ledger["source_commit"]
        ):
            raise ValueError(
                f"fit batch failed declaration/source/cache checks: {job['id']}"
            )
        for p in [
            gate_path,
            trials_path,
            folder / "trials.json",
            folder / "source-commit.txt",
        ]:
            inputs[str(p)] = digest(p)
        for trial in trials:
            name = trial["name"]
            if name in results or expected.get(name) != trial:
                raise ValueError("duplicate or undeclared fitting cell")
            fit_dir = folder / "fitting" / name
            record = audit_fit_artifacts(
                fit_dir, trial, cache_sha256=cache["cache_index_sha256"]
            )
            metadata = json.loads((fit_dir / "fit-complete.json").read_text())[
                "cache_extraction_metadata"
            ]
            if (
                any(
                    metadata["config"][key] != template[key]
                    for key in ["target", "proposer", "source_trunk"]
                )
                or metadata["relay_training"]["feature_cache"]
                != {"train_records": 2048, "validation_records": 1024}
                or trial["normalize_input"] != (args.family == "dflash")
            ):
                raise ValueError("fit interface or family normalization changed")
            checkpoints = {}
            for step in trial["checkpoint_steps"]:
                path = str(Path(job["fitting_output"]) / name / f"step-{step:06d}.pt")
                sha = gate["checkpoint_sha256"].get(path)
                if not isinstance(sha, str) or len(sha) != 64:
                    raise ValueError("GPU batch gate lacks a declared checkpoint")
                checkpoints[path] = sha
            for filename, sha in record["source_sha256"].items():
                inputs[str(fit_dir / filename)] = sha
            results[name] = {
                **record,
                "job": job["id"],
                "raw_directory": str(fit_dir),
                "batch_gate_path": str(gate_path),
                "checkpoint_sha256": checkpoints,
            }
        batches.append(
            {
                "job": job["id"],
                "fit_wall_seconds": gate["fit_wall_seconds"],
                "total_wall_seconds": gate["total_wall_seconds"],
            }
        )
    if results.keys() != expected.keys():
        raise ValueError("14B fitting matrix is incomplete")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "complete",
                "family": args.family,
                "target": template["target"],
                "primary_cells": len(matrix["primary_cells"]),
                "dense_seed_controls": len(matrix["dense_seed_controls"]),
                "matrix_sha256": digest(matrix_path),
                "feature_cache_index_sha256": cache["cache_index_sha256"],
                "input_sha256": inputs,
                "batches": batches,
                "results": results,
                "scope": "All 14 primary fits and two dense seed controls at 512/2048 examples, with complete per-update accounting and common validation. Raw GPU gates checked checkpoint tensors before collection; this local audit checks their recorded hash links and logs, not remote tensor contents. Endpoints remain fixed at 8192 updates regardless of validation minima. Decoding, answer quality, rate checks, and selected three-seed replication remain separate.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Audited all {len(results)} {args.family}14 fitting trajectories")


if __name__ == "__main__":
    main()
