"""Bounded end-to-end cache, four independent fits and shared-reference decoding."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch
import yaml

from relayspec.cache_data_inputs import checked_data_inputs


def run(command, env):
    subprocess.run(command, env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path(
            "configs/protocol_active/train_dflash_qwen3_8b_relative_4gpu.yaml"
        ),
    )
    parser.add_argument(
        "--campaign-config",
        type=Path,
        default=Path("configs/submission/scaling/campaign-pilot.yaml"),
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--training-manifest", default="train-32768.json")
    parser.add_argument("--report-domain-diagnostics", action="store_true")
    args = parser.parse_args()
    python = os.environ["RELAYSPEC_PYTHON"]
    artifacts = Path(os.environ["RELAYSPEC_OUTPUT"])
    cache_root = Path(os.environ["RELAYSPEC_FEATURE_CACHE"])
    fits = cache_root / "fits"
    artifacts.mkdir(parents=True, exist_ok=True)
    launcher = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
    ]
    train = yaml.safe_load(args.training_config.read_text())
    data_root = args.data_root or (
        Path(os.environ["RELAYSPEC_CACHE_DIR"]) / "relayspec/scaling-data/numina-v1"
    )
    data_inputs = checked_data_inputs(
        data_root,
        args.training_manifest,
        train_records=64,
        validation_records=16,
        report_domains=args.report_domain_diagnostics,
    )
    train["relay_training"].update(
        manifest_path=str(data_root / args.training_manifest),
        validation_manifest_path=str(data_root / "validation.json"),
        feature_cache={"train_records": 64, "validation_records": 16},
    )
    config_path = artifacts / "cache-config.yaml"
    config_path.write_text(yaml.safe_dump(train, sort_keys=False))
    run([*launcher, "scripts/train_relay.py", "--config", str(config_path)], os.environ)
    gate = json.loads((artifacts / "feature-cache-gate.json").read_text())
    if gate["status"] != "pass" or gate["counts"] != {"train": 64, "validation": 16}:
        raise ValueError("cache pilot failed its extraction gate")
    trials = []
    for name, architecture, width, l2 in [
        ("dense", "dense", None, 0),
        ("factor512", "factorized", 512, 0),
        ("mlp512", "mlp", 512, 0),
        ("mlp512-l2", "mlp", 512, 1e-6),
    ]:
        trial = {
            "name": name,
            "architecture": architecture,
            "seed": 1729,
            "steps": 16,
            "distinct_examples": 64,
            "learning_rate": 6e-4,
            "l2_weight": l2,
            "weight_decay": 0.0,
            "checkpoint_steps": [8, 16],
            "validation_records": 16,
        }
        if width is not None:
            trial["width"] = width
        if args.report_domain_diagnostics:
            trial.update(
                report_domain_diagnostics=True,
                validation_required_domains=sorted(
                    data_inputs["files"]["validation"]["domain_counts"]
                ),
            )
        trials.append(trial)
    trial_path = artifacts / "cached-trials.json"
    trial_path.write_text(json.dumps({"trials": trials}, indent=2) + "\n")
    fit_env = {**os.environ, "RELAYSPEC_OUTPUT": str(fits)}
    run(
        [
            *launcher,
            "scripts/fit_cached_mappers.py",
            "--trials",
            str(trial_path),
            "--equivalence-pilot",
        ],
        fit_env,
    )
    for trial in trials:
        path = fits / trial["name"]
        completion = json.loads((path / "fit-complete.json").read_text())
        if completion["status"] != "pass" or completion["steps"] != 16:
            raise ValueError("cached mapper fit did not finish")
        checkpoint = torch.load(
            path / "step-000016.pt", map_location="cpu", weights_only=True
        )
        if checkpoint["steps"] != 16 or any(
            not torch.isfinite(v).all() for v in checkpoint["relay"].values()
        ):
            raise ValueError("cached mapper checkpoint failed validation")
        del checkpoint
    campaign = yaml.safe_load(args.campaign_config.read_text())
    if (
        campaign["proposer"] != train["proposer"]
        or campaign["target"] != train["target"]
    ):
        raise ValueError("cache pilot training and decoding must use the same models")
    campaign["relay_probe"]["variants"] = {
        "relay_dense_a": str(fits / "dense/step-000016.pt"),
        "relay_factor512": str(fits / "factor512/step-000016.pt"),
        "relay_mlp512": str(fits / "mlp512/step-000016.pt"),
        "relay_dense_b": str(fits / "dense/step-000016.pt"),
        "relay_mlp512_l2": str(fits / "mlp512-l2/step-000016.pt"),
    }
    campaign["benchmark"]["methods"].append("relay_mlp512_l2")
    campaign_path = artifacts / "campaign-config.yaml"
    campaign_path.write_text(yaml.safe_dump(campaign, sort_keys=False))
    benchmark_dir = artifacts / "evaluation"
    run(
        [
            python,
            "scripts/run_mapper_campaign.py",
            "--config",
            str(campaign_path),
            "--equal-methods",
            "relay_dense_a",
            "relay_dense_b",
        ],
        {**os.environ, "RELAYSPEC_OUTPUT": str(benchmark_dir)},
    )
    for trial in trials:
        destination = artifacts / "fitting" / trial["name"]
        destination.mkdir(parents=True)
        for p in (fits / trial["name"]).iterdir():
            if p.suffix != ".pt":
                shutil.copy2(p, destination / p.name)
    (artifacts / "pilot-gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "data_inputs": data_inputs,
                "feature_cache": str(cache_root),
                "feature_cache_index_sha256": hashlib.sha256(
                    (cache_root / "cache-index.json").read_bytes()
                ).hexdigest(),
                "training_config_sha256": hashlib.sha256(
                    args.training_config.read_bytes()
                ).hexdigest(),
                "models": {"target": train["target"], "proposer": train["proposer"]},
                "fitted_maps": [t["name"] for t in trials],
                "evaluation": str(benchmark_dir),
                "scope": "Cache integrity, BF16 gradient equivalence within declared tolerances, independent four-GPU fitting, finite checkpoints, paired decoding and duplicate-map isolation. No full-answer quality or scaling claim.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
