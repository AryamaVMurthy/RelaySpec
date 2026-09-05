"""Declare paired decoding of three-seed small-data robustness panels."""

import argparse
import json
from pathlib import Path

import yaml
from summarize_focused_scaling import collect, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, inputs, cache_hash = {}, {}, None
    for matrix_name, registry_name in [
        ("matrix-focused-v1", "focused-results.json"),
        ("matrix-small-lr-v1", "learning-rate-results.json"),
        ("matrix-selected-small-seeds-v1", "selected-seed-results.json"),
    ]:
        matrix = Path("configs/submission/scaling") / matrix_name / "matrix.json"
        registry = Path("reports/mapper-scaling-20260905") / registry_name
        result = collect(args.raw_root, matrix)
        if result["status"] != "complete" or result != json.loads(registry.read_text()):
            raise ValueError("seed campaign requires complete source-verified fits")
        cache_hash = cache_hash or result["feature_cache_index_sha256"]
        if cache_hash != result["feature_cache_index_sha256"]:
            raise ValueError("seed fits use different features")
        if set(records) & set(result["results"]):
            raise ValueError("ambiguous seed fit source")
        records.update(result["results"])
        inputs.update({str(matrix): digest(matrix), str(registry): digest(registry)})
    variants, evidence = {}, {}
    for seed in (1729, 1730, 1731):
        names = [
            f"dense-n512-s{seed}",
            f"dense-n2048-s{seed}",
            f"factorized1024-n512-s{seed}",
            f"factorized4096-n2048-s{seed}-lr-0.0002",
            f"mlp4096-n2048-s{seed}-lr-0.0002",
            f"mlp4096-n512-s{seed}-lr-0.0002",
        ]
        for name in names:
            record = records[name]
            gate_path = args.raw_root / record["raw_directory"]
            gate_path = gate_path.parent.parent / "batch-gate.json"
            gate = json.loads(gate_path.read_text())
            for step in sorted({8192, record["best_validation_step"]}):
                filename = f"step-{step:06d}.pt"
                selected = [
                    (p, sha)
                    for p, sha in gate["checkpoint_sha256"].items()
                    if Path(p).parent.name == name and Path(p).name == filename
                ]
                if (
                    len(selected) != 1
                    or selected[0][1] != record["checkpoint_sha256"][filename]
                ):
                    raise ValueError("seed checkpoint is missing or inconsistent")
                alias = (
                    "relay_" + name.replace("-", "_").replace(".", "p") + f"_step{step}"
                )
                variants[alias] = selected[0][0]
                evidence[alias] = {
                    "trial": record["trial"],
                    "selected_step": step,
                    "best_validation_step": record["best_validation_step"],
                    "checkpoint_sha256": selected[0][1],
                    "fit_gate_sha256": digest(gate_path),
                    "validation_sha256": record["validation_sha256"],
                }
    config = yaml.safe_load(
        Path("configs/submission/scaling/campaign-pilot.yaml").read_text()
    )
    config["run_name"] = "selected-small-data-three-seed-decoding"
    config["benchmark"]["max_prompts"] = 16
    config["generation"]["max_new_tokens"] = 256
    config["relay_probe"]["variants"] = variants
    config["benchmark"]["methods"] = [
        "native_ar",
        "native_target_dflash",
        "optimized_source_reuse",
        *variants,
    ]
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "status": "declared_after_all_seed_fit_gates",
                "input_sha256": inputs,
                "feature_cache_index_sha256": cache_hash,
                "config_sha256": digest(args.output),
                "variants": evidence,
                "scope": "Six small-data fitting settings at three seeds.18 fixed8192-update "
                "endpoints plus every distinct minimum saved feature-validation checkpoint. "
                "Paired16-request256-token development decoding. Selection precedes decoding. "
                "Report endpoint and validation-selected seed variation separately. "
                "No full-answer quality claim or training-data expansion.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Declared {len(variants)} checkpoints from18 fits at three seeds")


if __name__ == "__main__":
    main()
