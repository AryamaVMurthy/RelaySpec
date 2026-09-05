"""Freeze decoding of wide-mapper rates and validation-selected early stops."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(raw_root, reports):
    sources = [
        reports / name
        for name in ("focused-results.json", "learning-rate-results.json")
    ]
    selected, cache_hash = {}, None
    for source in sources:
        registry = json.loads(source.read_text())
        if registry["status"] != "complete":
            raise ValueError("convergence decoding requires completed registries")
        cache_hash = cache_hash or registry["feature_cache_index_sha256"]
        if cache_hash != registry["feature_cache_index_sha256"]:
            raise ValueError("registries have different feature caches")
        for name, result in sorted(registry["results"].items()):
            trial = result["trial"]
            control = name in ("dense-n2048-s1729", "factorized1024-n512-s1729")
            if not control and trial.get("width") != 4096:
                continue
            if name in selected:
                raise ValueError("ambiguous convergence trial")
            folder = raw_root / result["raw_directory"]
            paths = {
                "fit_sha256": folder / "fit-complete.json",
                "validation_sha256": folder / "validation.jsonl",
                "batch_gate_sha256": folder.parent.parent / "batch-gate.json",
            }
            if any(digest(path) != result[key] for key, path in paths.items()):
                raise ValueError(f"raw evidence changed: {name}")
            fit = json.loads(paths["fit_sha256"].read_text())
            gate = json.loads(paths["batch_gate_sha256"].read_text())
            points = [
                json.loads(line)
                for line in paths["validation_sha256"].read_text().splitlines()
            ]
            if (
                fit["status"] != "pass"
                or gate["status"] != "pass"
                or gate["mode"] != "fit"
                or fit["trial"] != trial
                or trial["distinct_examples"] not in (512, 2048)
                or trial["steps"] != 8192
                or not trial["normalize_input"]
                or fit["feature_cache_index_sha256"] != cache_hash
                or gate["feature_cache_index_sha256"] != cache_hash
                or points != result["validation_trajectory"]
                or name not in gate["trials"]
            ):
                raise ValueError(f"inconsistent fit: {name}")
            best = min(
                points,
                key=lambda p: (p["groups"]["validation"]["objective"], p["step"]),
            )["step"]
            if best != result["best_validation_step"]:
                raise ValueError("recorded best step differs from raw validation")
            steps = [8192] if control else sorted({8192, best})
            variants = []
            for step in steps:
                filename = f"step-{step:06d}.pt"
                checkpoints = [
                    (p, sha)
                    for p, sha in gate["checkpoint_sha256"].items()
                    if Path(p).parent.name == name and Path(p).name == filename
                ]
                if (
                    len(checkpoints) != 1
                    or result["checkpoint_sha256"].get(filename) != checkpoints[0][1]
                ):
                    raise ValueError(
                        f"missing or inconsistent selected checkpoint: {name}/{step}"
                    )
                alias = (
                    "relay_"
                    + name.replace("-s1729", "").replace("-", "_").replace(".", "p")
                    + f"_step{step}"
                )
                variants.append(
                    (
                        alias,
                        checkpoints[0][0],
                        {
                            "trial": trial,
                            "step": step,
                            "selection": "fixed8192"
                            if step == 8192
                            else "minimum_development_validation",
                            "best_validation_step": best,
                            "checkpoint_sha256": checkpoints[0][1],
                            "registry_sha256": digest(source),
                            "raw_directory": result["raw_directory"],
                            **{key: result[key] for key in paths},
                        },
                    )
                )
            selected[name] = variants
    if len(selected) != 14:
        raise ValueError("expected12 wide rate trials and two deployment controls")
    variants = [variant for group in selected.values() for variant in group]
    return cache_hash, variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--reports", type=Path, default=Path("reports/mapper-scaling-20260905")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache_hash, variants = build(args.raw_root, args.reports)
    template = Path("configs/submission/scaling/campaign-pilot.yaml")
    config = yaml.safe_load(template.read_text())
    config["run_name"] = "small-data-learning-rate-and-early-stop-decoding"
    config["benchmark"]["max_prompts"] = 16
    config["generation"]["max_new_tokens"] = 256
    config["relay_probe"]["variants"] = {name: path for name, path, _ in variants}
    config["benchmark"]["methods"] = [
        "native_ar",
        "native_target_dflash",
        "optimized_source_reuse",
        *config["relay_probe"]["variants"],
    ]
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "status": "declared_before_convergence_decoding",
                "feature_cache_index_sha256": cache_hash,
                "config_sha256": digest(args.output),
                "template_sha256": digest(template),
                "variants": {name: evidence for name, _, evidence in variants},
                "scope": "Width4096 factorized/MLP, N512/2048, learning rates2e-4/6e-4/1.8e-3. "
                "Every8192-update endpoint plus every distinct minimum-validation checkpoint. "
                "Two smaller/deployment controls and three inherited references. "
                "Paired16-request256-token development decoding, no full-answer quality claim. "
                "Early stopping is selected using feature validation, not these decoding outputs. "
                "No additional fitting or large-data expansion.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Declared {len(variants)} mapper checkpoints plus three references")


if __name__ == "__main__":
    main()
