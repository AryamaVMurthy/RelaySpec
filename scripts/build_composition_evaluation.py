"""Bind predeclared domain-composition evaluation to audited endpoint checkpoints."""

import copy
import json
from pathlib import Path

import yaml

from relayspec.cached_fit_evidence import digest


def main():
    base = Path("configs/submission/scaling/composition-small-v1")
    root = Path("reports/mapper-scaling-20260905/resumed-composition")
    declaration = json.loads((base / "evaluation-protocol.json").read_text())
    for path, sha in declaration["manifests_sha256"].items():
        assert digest(Path(path)) == sha
    jobs = json.loads((root / "jobs.json").read_text())
    variants = {}
    inputs = {
        str(base / "evaluation-protocol.json"): digest(
            base / "evaluation-protocol.json"
        )
    }
    for arm in ["math", "mixed"]:
        audit_path = root / f"{arm}-fits-audit.json"
        audit = json.loads(audit_path.read_text())
        assert audit["status"] == "complete" and audit["stage"] == "fits"
        for path, sha in audit["input_sha256"].items():
            assert digest(Path(path)) == sha
        inputs[str(audit_path)] = digest(audit_path)
        job = next(j for j in jobs["jobs"] if j["arm"] == arm and j["stage"] == "fits")
        gate_path = root / f"run-{job['id']}" / "batch-gate.json"
        gate = json.loads(gate_path.read_text())
        inputs[str(gate_path)] = digest(gate_path)
        for trial in json.loads((base / f"{arm}-fits.json").read_text())["trials"]:
            matches = [
                (p, sha)
                for p, sha in gate["checkpoint_sha256"].items()
                if Path(p).parent.name == trial["name"]
                and Path(p).name == "step-008192.pt"
            ]
            assert len(matches) == 1
            p, sha = matches[0]
            variants["relay_" + trial["name"].replace("-", "_")] = {
                "trial": trial,
                "checkpoint": p,
                "checkpoint_sha256": sha,
            }
    assert len(variants) == 8
    config = yaml.safe_load(
        Path("configs/submission/scaling/campaign-pilot.yaml").read_text()
    )
    config["relay_probe"]["variants"] = {
        k: v["checkpoint"] for k, v in variants.items()
    }
    config["benchmark"]["methods"] = [
        "native_ar",
        "native_target_dflash",
        "optimized_source_reuse",
        *variants,
    ]
    config["benchmark"]["benchmarks"] = declaration["tasks"]
    stages = {}
    for stage, settings in declaration["stages"].items():
        cfg = copy.deepcopy(config)
        manifest = base / f"evaluation-{stage}-manifest.json"
        records = json.loads(manifest.read_text())["records"]
        cfg["run_name"] = f"composition-evaluation-{stage}"
        cfg["benchmark"].update(manifest_path=str(manifest), max_prompts=len(records))
        cfg["generation"]["max_new_tokens"] = settings["max_new_tokens"]
        path = base / f"evaluation-{stage}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        stages[stage] = {
            "config": str(path),
            "config_sha256": digest(path),
            "manifest": str(manifest),
            "manifest_sha256": digest(manifest),
            "requests": len(records),
            "problem_ids": [r["problem_id"] for r in records],
        }
    (base / "evaluation-bound-protocol.json").write_text(
        json.dumps(
            {
                "status": "all_endpoints_bound_after_fit_audit",
                "input_sha256": inputs,
                "methods": config["benchmark"]["methods"],
                "variants": variants,
                "stages": stages,
                "declaration": declaration,
            },
            indent=2,
        )
        + "\n"
    )
    print("Bound eight endpoints, pilot and full evaluation")


if __name__ == "__main__":
    main()
