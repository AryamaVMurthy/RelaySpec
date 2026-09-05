"""Declare full-answer development evaluation from verified small-data fits."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # Selection uses existing development results. These are not held-out tests.
    selected = {
        "dense-n512-s1729": 8192,
        "dense-n2048-s1729": 8192,
        "factorized1024-n512-s1729": 8192,
        "factorized4096-n2048-s1729": 8192,
        "mlp4096-n2048-s1729": 8192,
        "factorized512-n2048-s1729": 8192,
        "mlp512-n2048-s1729": 8192,
        "factorized512-n2048-s1729-continue32768": 32768,
        "mlp512-n2048-s1729-continue32768": 32768,
    }
    variants, provenance = {}, {}
    cache_hash = None
    for path in sorted(
        (args.raw_root / "reports/mapper-scaling-20260905").glob(
            "*/run-*/batch-gate.json"
        )
    ):
        gate = json.loads(path.read_text())
        if gate.get("status") != "pass" or gate.get("mode") != "fit":
            continue
        for name in set(gate["trials"]) & selected.keys():
            alias = "relay_" + name.replace("-s1729", "").replace("-", "_")
            if alias in variants:
                raise ValueError(f"ambiguous completed trial: {name}")
            fit_path = path.parent / "fitting" / name / "fit-complete.json"
            fit = json.loads(fit_path.read_text())
            trial = fit["trial"]
            cache_hash = cache_hash or gate["feature_cache_index_sha256"]
            if (
                fit["status"] != "pass"
                or trial["name"] != name
                or trial["distinct_examples"] not in (512, 2048)
                or trial["steps"] != selected[name]
                or fit["feature_cache_index_sha256"] != cache_hash
                or gate["feature_cache_index_sha256"] != cache_hash
            ):
                raise ValueError(f"inconsistent completed small-data fit: {name}")
            checkpoints = [
                (p, sha)
                for p, sha in gate["checkpoint_sha256"].items()
                if Path(p).parent.name == name
                and Path(p).name == f"step-{selected[name]:06d}.pt"
            ]
            if len(checkpoints) != 1:
                raise ValueError(f"missing or ambiguous checkpoint: {name}")
            variants[alias] = checkpoints[0][0]
            provenance[alias] = {
                "trial": trial,
                "checkpoint_sha256": checkpoints[0][1],
                "fit_sha256": digest(fit_path),
                "gate_sha256": digest(path),
                "raw_gate": str(path.relative_to(args.raw_root)),
            }
    if len(variants) != len(selected):
        raise ValueError("selected small-data checkpoints are incomplete")
    # Preserve declared order so rebuilding produces identical artifacts.
    ordered = ["relay_" + n.replace("-s1729", "").replace("-", "_") for n in selected]
    variants = {name: variants[name] for name in ordered}
    config = yaml.safe_load(
        Path("configs/submission/scaling/campaign-pilot.yaml").read_text()
    )
    config["generation"]["max_new_tokens"] = 2048
    config["relay_probe"]["variants"] = variants
    config["benchmark"]["methods"] = [
        "native_ar",
        "native_target_dflash",
        "optimized_source_reuse",
        *variants,
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for stage, count in [("pilot", 4), ("development", 128)]:
        config["run_name"] = f"small-data-full-answer-{stage}"
        config["benchmark"]["max_prompts"] = count
        path = args.output_dir / f"campaign-small-data-quality-{stage}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        hashes[path.name] = digest(path)
    (args.output_dir / "small-data-quality.provenance.json").write_text(
        json.dumps(
            {
                "status": "predeclared_from_completed_fits",
                "feature_cache_index_sha256": cache_hash,
                "configs_sha256": hashes,
                "manifest_sha256": digest(Path(config["benchmark"]["manifest_path"])),
                "variants": {name: provenance[name] for name in ordered},
                "scope": "Development MATH evaluation, greedy decoding with a 2048-token cap. "
                "Nine small-data mappers and three shared controls. Includes matched "
                "8192 versus 32768 update endpoints on the same 2048 distinct examples. "
                "Pilot four requests under ten minutes; full 128-request run only after "
                "pilot completion and resource review. Report cap hits separately. "
                "Existing development prompts are exposed and do not constitute "
                "untouched confirmation. No fitting or large-data expansion.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Declared {len(variants)} selected mappers plus three controls")


if __name__ == "__main__":
    main()
