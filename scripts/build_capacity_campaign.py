"""Require every primary fit gate before declaring paired capacity decoding."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from relayspec.mapper_campaign import campaign_references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("configs/submission/scaling/matrix-focused-v1/matrix.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-endpoints", action="store_true")
    parser.add_argument("--include-seed-controls", action="store_true")
    parser.add_argument("--fit-registry", type=Path)
    parser.add_argument(
        "--checkpoint-selection", choices=("endpoint", "validation"), default="endpoint"
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("configs/submission/scaling/campaign-pilot.yaml"),
    )
    args = parser.parse_args()
    if args.include_endpoints and args.checkpoint_selection != "validation":
        raise ValueError("paired endpoints require validation-selected checkpoints")
    matrix = json.loads(args.matrix.read_text())
    primary = matrix["primary_cells"]
    if args.include_seed_controls:
        primary = primary + matrix["dense_seed_controls"]
    gates = {}
    if args.fit_registry:
        registry = json.loads(args.fit_registry.read_text())
        if (
            registry.get("status") != "complete"
            or registry["matrix_sha256"]
            != hashlib.sha256(args.matrix.read_bytes()).hexdigest()
        ):
            raise ValueError("completed fitting registry must match the matrix")
        for name, sha in registry["input_sha256"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
                raise ValueError("audited fitting source changed")
        if set(registry["results"]) != {t["name"] for t in primary}:
            raise ValueError("campaign must include every audited fitting cell")
        gate_paths = sorted(
            {Path(r["batch_gate_path"]) for r in registry["results"].values()}
        )
    else:
        gate_paths = (args.raw_root / "reports/mapper-scaling-20260905").glob(
            "*/run-*/batch-gate.json"
        )
    for p in gate_paths:
        g = json.loads(p.read_text())
        if g.get("status") != "pass" or g.get("mode") != "fit":
            continue
        for name in g["trials"]:
            if name in gates:
                raise ValueError(
                    f"multiple completed fits need explicit resolution: {name}"
                )
            gates[name] = (p, g)
    config = yaml.safe_load(args.template.read_text())
    references = campaign_references(
        config["proposer"]["family"], config["benchmark"]["methods"]
    )
    config["run_name"] = "small-data-complete-capacity-decoding"
    if args.checkpoint_selection == "validation":
        config["run_name"] = "small-data-validation-selected-capacity-decoding"
    config["generation"]["max_new_tokens"] = 256
    config["benchmark"]["max_prompts"] = 16
    variants, provenance = {}, {}
    cache_hash = None
    for t in primary:
        if t["name"] not in gates:
            raise ValueError(f"primary fit is incomplete: {t['name']}")
        p, g = gates[t["name"]]
        if (
            "campaign_config_sha256" in g
            and g["campaign_config_sha256"]
            != hashlib.sha256(args.template.read_bytes()).hexdigest()
        ):
            raise ValueError(
                "fit gate was validated with a different family campaign template"
            )
        cache_hash = cache_hash or g["feature_cache_index_sha256"]
        if cache_hash != g["feature_cache_index_sha256"]:
            raise ValueError("primary fits have different frozen features")
        step, validation_path = 8192, None
        if args.checkpoint_selection == "validation":
            validation_path = p.parent / "fitting" / t["name"] / "validation.jsonl"
            points = [
                json.loads(line) for line in validation_path.read_text().splitlines()
            ]
            if {point["step"] for point in points} != {0, *t["checkpoint_steps"]}:
                raise ValueError(
                    "validation selection requires every declared checkpoint"
                )
            step = min(
                points,
                key=lambda point: (
                    point["groups"]["validation"]["objective"],
                    point["step"],
                ),
            )["step"]
        paths = [
            (k, sha)
            for k, sha in g["checkpoint_sha256"].items()
            if Path(k).parent.name == t["name"]
            and Path(k).name == f"step-{step:06d}.pt"
        ]
        if len(paths) != 1:
            raise ValueError("missing or ambiguous selected checkpoint")
        complete = json.loads(
            (p.parent / "fitting" / t["name"] / "fit-complete.json").read_text()
        )
        comparable = {
            k: v
            for k, v in complete["trial"].items()
            if k not in {"cache_backend", "device_cache"}
        }
        declared = {
            k: v for k, v in t.items() if k not in {"cache_backend", "device_cache"}
        }
        if comparable != declared:
            raise ValueError("completed fit does not match its declared primary cell")
        expected_normalization = config["proposer"]["family"] == "dflash"
        if t.get("normalize_input", expected_normalization) != expected_normalization:
            raise ValueError(
                "declared mapper normalization differs from this family protocol"
            )
        alias = "relay_" + t["name"].replace("-s1729", "").replace("-", "_")
        variants[alias] = paths[0][0]
        provenance[alias] = {
            "trial": t,
            "checkpoint_sha256": paths[0][1],
            "fit_gate_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        if validation_path:
            provenance[alias].update(
                selected_step=step,
                selection="minimum_feature_validation_before_decoding",
                validation_sha256=hashlib.sha256(
                    validation_path.read_bytes()
                ).hexdigest(),
            )
        if args.include_endpoints and step != 8192:
            endpoints = [
                (k, sha)
                for k, sha in g["checkpoint_sha256"].items()
                if Path(k).parent.name == t["name"] and Path(k).name == "step-008192.pt"
            ]
            if len(endpoints) != 1:
                raise ValueError("missing matched endpoint for early stopping")
            variants[alias + "_endpoint"] = endpoints[0][0]
            provenance[alias + "_endpoint"] = {
                **provenance[alias],
                "selected_step": 8192,
                "selection": "fixed_exposure_endpoint",
                "checkpoint_sha256": endpoints[0][1],
            }
    config["relay_probe"]["variants"] = variants
    config["benchmark"]["methods"] = [*references, *variants]
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "status": "declared_after_all_primary_fit_gates",
                "matrix_sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
                **(
                    {
                        "fit_registry_sha256": hashlib.sha256(
                            args.fit_registry.read_bytes()
                        ).hexdigest(),
                        "primary_cells": len(matrix["primary_cells"]),
                        "dense_seed_controls": len(matrix["dense_seed_controls"])
                        if args.include_seed_controls
                        else 0,
                    }
                    if args.fit_registry
                    else {}
                ),
                "config_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "feature_cache_index_sha256": cache_hash,
                "variants": provenance,
                "scope": (
                    f"{len(primary)} predeclared primary endpoints at8192updates, N512/2048. Paired16-request256-token development decoding; neither full-answer quality nor isolated deployment memory."
                    if args.checkpoint_selection == "endpoint"
                    else f"{len(primary)} primary mappers at their minimum saved feature-validation checkpoint, N512/2048. Selection precedes decoding. {len(variants)} total checkpoints including any requested distinct fixed-exposure endpoints. Paired16-request256-token development decoding; neither full-answer quality nor isolated deployment memory."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Declared {len(variants)} mapper checkpoints with shared references")


if __name__ == "__main__":
    main()
