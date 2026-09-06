"""Synthetic evidence fixtures exercise binding failures before GPU results arrive."""

import json
import runpy
import sys
from pathlib import Path

import pytest
import yaml

from relayspec.cached_fit_evidence import digest


@pytest.mark.parametrize(
    "damage", [None, "missing_method", "checkpoint", "answer", "fit_gate", "source"]
)
def test_capacity_audit_requires_exact_endpoints_and_request_evidence(
    tmp_path, monkeypatch, damage
):
    script = Path("scripts/audit_target14_capacity.py").resolve()
    matrix = json.loads(
        Path(
            "configs/submission/scaling/target14b-small-v1/matrix-dflash/matrix.json"
        ).read_text()
    )
    trials = matrix["primary_cells"] + matrix["dense_seed_controls"]
    monkeypatch.chdir(tmp_path)

    def write(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    gate = write("fit-gate.json", {"fixture": "synthetic fit gate"})
    variants, fits, paths = {}, {}, {}
    for i, trial in enumerate(trials):
        method, checkpoint, sha = (
            f"relay_test{i}",
            f"/scratch/fixture/{trial['name']}/step-008192.pt",
            f"{i:064x}",
        )
        variants[method] = {
            "trial": trial,
            "checkpoint_sha256": sha,
            "fit_gate_sha256": digest(gate),
        }
        paths[method] = checkpoint
        fits[trial["name"]] = {
            "trial": trial,
            "checkpoint_sha256": {checkpoint: sha},
            "batch_gate_path": str(gate),
        }
    target = {"id": "fixture-14B", "revision": "fixture-revision"}
    registry = write(
        "reports/mapper-scaling-20260905/target14b-dflash-fit-results.json",
        {
            "status": "complete",
            "family": "dflash",
            "input_sha256": {str(gate): digest(gate)},
            "matrix_sha256": "matrix",
            "feature_cache_index_sha256": "cache",
            "results": fits,
            "target": target,
        },
    )
    manifest = write(
        "manifest.json",
        {
            "records": [
                {"problem_id": f"p{i}", "benchmark": "math500", "answer": "42"}
                for i in range(16)
            ]
        },
    )
    methods = ["native_ar", "optimized_source_reuse", *variants]
    config = {
        "target": target,
        "proposer": {"family": "dflash"},
        "seed": 1729,
        "generation": {"max_new_tokens": 256},
        "benchmark": {
            "manifest_path": str(manifest),
            "methods": methods,
            "max_prompts": 16,
            "benchmarks": ["math500"],
        },
        "relay_probe": {"repetitions": 1, "variants": paths},
    }
    config_path = Path(
        "configs/submission/scaling/target14b-small-v1/campaign-dflash-capacity.yaml"
    )
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config))
    write(
        config_path.with_suffix(".provenance.json"),
        {
            "fit_registry_sha256": digest(registry),
            "config_sha256": digest(config_path),
            "matrix_sha256": "matrix",
            "feature_cache_index_sha256": "cache",
            "primary_cells": 14,
            "dense_seed_controls": 2,
            "variants": variants,
        },
    )
    run = Path("run-fixture")
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    (run / "source-commit.txt").write_text("fixture-source")
    ledger = write(
        "jobs.json",
        {
            "source_commit": "fixture-source",
            "jobs": [{"id": 1, "family": "dflash", "local": str(run)}],
        },
    )
    rows = [
        {
            "problem_id": f"p{i}",
            "repetition": 0,
            "method": m,
            "benchmark": "math500",
            "reference_answer": "42",
            "correct": True,
            "output_tokens": 256,
            "request_seconds": 1.0,
            "output_hash": "fixture-output",
            "acceptance_length": 1.0,
            **(
                {"mapper_checkpoint_sha256": variants[m]["checkpoint_sha256"]}
                if m in variants
                else {}
            ),
        }
        for i in range(16)
        for m in methods
    ]
    write(
        run / "completion-gate.json",
        {"status": "pass", "records": len(rows), "requests": 16},
    )
    write(run / "campaign-gate.json", {"status": "pass", "methods": methods})
    write(run / "analysis.json", {})
    for relative in [
        "reports/ar-revision-20260905/scorer-provenance.json",
        "scripts/analyze_controlled_run.py",
        "src/relayspec/ar_paper_evidence.py",
    ]:
        write(Path("scorer") / relative, {"fixture": True})
    if damage == "missing_method":
        rows.pop()
    elif damage == "checkpoint":
        rows[-1]["mapper_checkpoint_sha256"] = "wrong"
    elif damage == "answer":
        rows[-1]["reference_answer"] = "wrong"
    elif damage == "fit_gate":
        gate.write_text("changed")
    elif damage == "source":
        (run / "source-commit.txt").write_text("changed")
    for name in ["benchmark-rank0.jsonl", "math-scored.jsonl"]:
        (run / name).write_text("\n".join(map(json.dumps, rows)))
    replays = []
    monkeypatch.setattr(
        "relayspec.quality_scoring.verify_saved_scores",
        lambda *args: replays.append(args),
    )
    output = Path("result.json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--family",
            "dflash",
            "--run",
            str(run),
            "--ledger",
            str(ledger),
            "--scoring-repo",
            "scorer",
            "--output",
            str(output),
        ],
    )
    if damage is None:
        runpy.run_path(str(script), run_name="__main__")
        result = json.loads(output.read_text())
        assert result["records"] == 288
        assert len(result["variants"]) == 16
        assert len(replays) == 1
    else:
        with pytest.raises(ValueError):
            runpy.run_path(str(script), run_name="__main__")
        assert not output.exists()
        assert not replays
