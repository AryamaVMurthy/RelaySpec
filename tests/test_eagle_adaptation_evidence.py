import json
import runpy
import sys
from pathlib import Path

import pytest
import yaml

from relayspec.cached_fit_evidence import digest


def test_family_interface_proofs_do_not_substitute_for_each_other():
    module = runpy.run_path("scripts/audit_adaptation_calibration.py")
    check = module["check_interface_proof"]
    dflash = {"direct_fusion_equivalence": {"status": "pass"}}
    eagle = {
        "eagle_initial_cache_equivalence": [
            {"status": "pass", "logits_bit_identical": True} for _ in range(4)
        ]
    }
    check(dflash, "dflash")
    check(eagle, "eagle3")
    with pytest.raises(ValueError, match="EAGLE"):
        check(dflash, "eagle3")
    eagle["eagle_initial_cache_equivalence"][2]["logits_bit_identical"] = False
    with pytest.raises(ValueError, match="EAGLE"):
        check(eagle, "eagle3")
    assert module["family_evidence_fields"]("eagle3") == (
        "checkpoint_sha256",
        "drafter_update_sha256",
        ("acceptance_lengths", "target_calls", "draft_calls"),
    )
    with pytest.raises(ValueError, match="unsupported"):
        module["family_evidence_fields"]("unknown")


def test_calibration_builder_requires_unchanged_audited_pilot(tmp_path, monkeypatch):
    script = Path("scripts/build_eagle_adaptation_calibration.py").resolve()
    relative = Path("configs/submission/scaling/eagle3-adaptation-pilot.yaml")
    original = relative.read_text()
    monkeypatch.chdir(tmp_path)
    relative.parent.mkdir(parents=True)
    relative.write_text(original)
    ledger = Path("reports/mapper-scaling-20260905/eagle3-adaptation-pilot/jobs.json")
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {"jobs": [{"local": "run-fixture", "remote": "/remote/run-fixture"}]}
        )
    )
    gate = Path("run-fixture/adaptation-pilot-gate.json")
    gate.parent.mkdir()
    gate.write_text(json.dumps({"config_sha256": digest(relative)}))
    pilot = Path("pilot-result.json")
    pilot.write_text(
        json.dumps(
            {
                "status": "complete",
                "family": "eagle3",
                "stage": "compatibility_pilot",
                "input_sha256": {str(p): digest(p) for p in [relative, ledger, gate]},
            }
        )
    )
    output = Path("calibration.yaml")
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "--pilot-result", str(pilot), "--output", str(output)],
    )
    runpy.run_path(str(script), run_name="__main__")
    config = yaml.safe_load(output.read_text())
    settings = config["adaptation_pilot"]
    prior = yaml.safe_load(original)["adaptation_pilot"]
    assert settings["pilot_distinct_examples"] == 512
    assert settings["pilot_updates"] == 128
    assert settings["initial_mapper_sha256"] == prior["initial_mapper_sha256"]
    assert settings["feature_cache_index_sha256"] == prior["feature_cache_index_sha256"]
    assert settings["prerequisite_gate"] == {
        "path": "/remote/run-fixture/adaptation-pilot-gate.json",
        "sha256": digest(gate),
    }
    assert settings["prepared_records_device"] == "cpu"
    assert settings["check_direct_fusion_equivalence"] is False
    output.unlink()
    gate.write_text(gate.read_text() + "\n")
    with pytest.raises(ValueError, match="evidence changed"):
        runpy.run_path(str(script), run_name="__main__")
    assert not output.exists()
