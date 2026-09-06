import hashlib
import json
import runpy
import sys
from pathlib import Path

import pytest
import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_audited_campaign_keeps_seed_controls_and_rejects_changed_evidence(
    tmp_path, monkeypatch
):
    matrix = tmp_path / "matrix.json"
    cells = [
        {
            "name": f"dense-n512-s{seed}",
            "architecture": "dense",
            "seed": seed,
            "distinct_examples": 512,
            "normalize_input": True,
        }
        for seed in [1729, 1730]
    ]
    matrix.write_text(
        json.dumps({"primary_cells": cells[:1], "dense_seed_controls": cells[1:]})
    )
    template = Path("configs/submission/scaling/campaign-pilot.yaml").resolve()
    gate = tmp_path / "run/batch-gate.json"
    gate.parent.mkdir()
    checkpoints = {f"/scratch/test/{t['name']}/step-008192.pt": "a" * 64 for t in cells}
    gate.write_text(
        json.dumps(
            {
                "status": "pass",
                "mode": "fit",
                "trials": [t["name"] for t in cells],
                "campaign_config_sha256": sha(template),
                "feature_cache_index_sha256": "cache",
                "checkpoint_sha256": checkpoints,
            }
        )
    )
    inputs = {
        str(gate): sha(gate),
        str(matrix): sha(matrix),
        str(template): sha(template),
    }
    for cell in cells:
        path = gate.parent / "fitting" / cell["name"] / "fit-complete.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"trial": cell}))
        inputs[str(path)] = sha(path)
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "status": "complete",
                "matrix_sha256": sha(matrix),
                "input_sha256": inputs,
                "results": {t["name"]: {"batch_gate_path": str(gate)} for t in cells},
            }
        )
    )
    output = tmp_path / "campaign.yaml"
    command = [
        "build_capacity_campaign.py",
        "--raw-root",
        str(tmp_path),
        "--matrix",
        str(matrix),
        "--output",
        str(output),
        "--template",
        str(template),
        "--fit-registry",
        str(registry),
    ]
    script = Path("scripts/build_capacity_campaign.py").resolve()
    monkeypatch.setattr(sys, "argv", command)
    with pytest.raises(ValueError, match="every audited fitting cell"):
        runpy.run_path(str(script), run_name="__main__")
    assert not output.exists()
    monkeypatch.setattr(sys, "argv", [*command, "--include-seed-controls"])
    runpy.run_path(str(script), run_name="__main__")
    config = yaml.safe_load(output.read_text())
    assert len(config["relay_probe"]["variants"]) == 2
    assert config["generation"]["max_new_tokens"] == 256
    assert config["benchmark"]["max_prompts"] == 16
    assert "relay_dense_n512_s1730" in config["benchmark"]["methods"]
    provenance = json.loads(output.with_suffix(".provenance.json").read_text())
    assert provenance["primary_cells"] == provenance["dense_seed_controls"] == 1
    assert provenance["fit_registry_sha256"] == sha(registry)
    gate.write_text(gate.read_text() + "\n")
    with pytest.raises(ValueError, match="source changed"):
        runpy.run_path(str(script), run_name="__main__")
