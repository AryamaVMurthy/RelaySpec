"""Run one controlled fitting job and paired evaluations within its allocation."""

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import torch
import yaml

from relayspec.scaling_trials import validate_trial


def run(command, env):
    subprocess.run(command, env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=[
            "pilot",
            "continuous",
            "data64",
            "data128",
            "data256",
            "data512",
            "data1024",
            "data2048",
        ],
        default="pilot",
    )
    parser.add_argument("--trial-config", type=Path)
    args = parser.parse_args()
    trial = json.loads(args.trial_config.read_text()) if args.trial_config else None
    budget = validate_trial(trial) if trial else None
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    python = os.environ["RELAYSPEC_PYTHON"]
    launcher = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
    ]
    pilot = args.experiment == "pilot"
    distinct = (
        int(args.experiment.removeprefix("data"))
        if args.experiment.startswith("data")
        else 4096
    )
    steps = 16 if pilot else 2048 if args.experiment == "continuous" else 1024
    saves = (
        [8, 16]
        if pilot
        else [128, 256, 512, 1024, 2048]
        if args.experiment == "continuous"
        else [1024]
    )
    train = yaml.safe_load(
        Path(
            "configs/protocol_active/train_dflash_qwen3_8b_relative_4gpu.yaml"
        ).read_text()
    )
    if trial:
        distinct = int(trial["distinct_examples"])
        steps = budget["steps"]
        saves = [steps]
        pilot = trial["stage"] == "pilot"
        train["seed"] = int(trial["seed"])
        train["relay_training"].update(
            factorized_rank=trial.get("width")
            if trial["architecture"] == "factorized"
            else None,
            mlp_hidden_width=trial.get("width")
            if trial["architecture"] == "mlp"
            else None,
            l2_weight=float(trial.get("l2_weight", 0.0)),
            weight_decay=float(trial.get("weight_decay", 0.0)),
        )
        (output / "trial.json").write_text(
            json.dumps(
                {
                    "spec": trial,
                    "budget": budget,
                    "config_sha256": hashlib.sha256(
                        args.trial_config.read_bytes()
                    ).hexdigest(),
                },
                indent=2,
            )
            + "\n"
        )
    train["run_name"] = (
        f"controlled-{trial['name'] if trial else args.experiment}-{os.environ['SLURM_JOB_ID']}"
    )
    manifest = json.loads(Path(train["relay_training"]["manifest_path"]).read_text())
    if trial:
        if distinct + 32 > len(manifest["records"]):
            raise ValueError(
                "trial requires 32 disjoint fitting-validation records beyond the fit budget"
            )
        validation_path = output / "validation-manifest.json"
        validation_path.write_text(
            json.dumps(
                {
                    "records": manifest["records"][-32:],
                    "role": "fitting validation for this trial; historically exposed training pool",
                },
                indent=2,
            )
            + "\n"
        )
        train["relay_training"]["validation_manifest_path"] = str(validation_path)
    if distinct > len(manifest["records"]):
        raise ValueError("declared distinct budget exceeds available manifest")
    manifest["records"] = manifest["records"][:distinct]
    manifest_path = output / "fit-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    train["relay_training"].update(
        manifest_path=str(manifest_path),
        distinct_examples=distinct,
        fit_examples=steps * 4,
        steps=steps,
        checkpoint_steps=saves,
    )
    train_path = output / "train.yaml"
    train_path.write_text(yaml.safe_dump(train, sort_keys=False))
    train_output = output / "training"
    env = {**os.environ, "RELAYSPEC_OUTPUT": str(train_output)}
    run([*launcher, "scripts/train_relay.py", "--config", str(train_path)], env)
    summary = json.loads((train_output / "relay-training-summary.json").read_text())
    if summary["status"] != "complete" or summary["steps"] != steps:
        raise ValueError("training did not reach the declared budget")
    for rank in range(4):
        rows = [
            json.loads(line)
            for line in (train_output / f"relay-train-rank{rank}.jsonl")
            .read_text()
            .splitlines()
        ]
        if [r["step"] for r in rows] != list(range(1, steps + 1)) or any(
            not math.isfinite(r["loss"]) or not math.isfinite(r["gradient_norm"])
            for r in rows
        ):
            raise ValueError(f"invalid training trace for rank {rank}")
    checkpoint_dir = Path(summary["checkpoint"]).parent
    checks = []
    previous_hash = None
    for step in saves:
        path = checkpoint_dir / f"step-{step:06d}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["steps"] != step or not saved["optimizer"]["state"]:
            raise ValueError("checkpoint state missing")
        if any(
            int(state["step"]) != step for state in saved["optimizer"]["state"].values()
        ):
            raise ValueError("optimizer steps do not match continuous trajectory")
        if any(not torch.isfinite(value).all() for value in saved["relay"].values()):
            raise ValueError("nonfinite relay weights")
        weights_hash = hashlib.sha256(
            b"".join(value.numpy().tobytes() for value in saved["relay"].values())
        ).hexdigest()
        if previous_hash == weights_hash:
            raise ValueError("successive snapshots have identical weights")
        previous_hash = weights_hash
        portable = output / f"relay-step-{step:06d}.pt"
        torch.save({k: v for k, v in saved.items() if k != "optimizer"}, portable)
        checks.append(
            {
                "step": step,
                "weights_sha256": weights_hash,
                "optimizer_step_verified": True,
                "checkpoint": str(portable),
                "data_accounting": saved["data_accounting"],
                "training_seconds": saved["elapsed_training_seconds"],
            }
        )
        del saved
    (output / "checkpoint-gate.json").write_text(
        json.dumps({"status": "pass", "checkpoints": checks}, indent=2) + "\n"
    )
    benchmark = yaml.safe_load(
        Path("configs/submission/day1/dflash-trace8.yaml").read_text()
    )
    benchmark["benchmark"].update(
        manifest_path="configs/eval_manifest.json",
        max_prompts=8 if pilot else 128,
        warmups=1,
    )
    benchmark["generation"]["max_new_tokens"] = 64 if pilot else 2048
    if trial:
        benchmark["benchmark"]["max_prompts"] = budget["max_prompts"]
        benchmark["benchmark"]["profile_regions"] = False
        benchmark["generation"]["max_new_tokens"] = budget["max_new_tokens"]
    for check in checks:
        config = copy.deepcopy(benchmark)
        config["run_name"] = train["run_name"] + f"-step{check['step']}"
        config["relay_probe"]["checkpoint_path"] = check["checkpoint"]
        config_path = output / f"benchmark-step-{check['step']:06d}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        bench_output = output / f"evaluation-step-{check['step']:06d}"
        bench_output.mkdir()
        shutil.copy2(config_path, bench_output / "config.yaml")
        env = {**os.environ, "RELAYSPEC_OUTPUT": str(bench_output)}
        run(
            [*launcher, "scripts/benchmark_relay.py", "--config", str(config_path)], env
        )
        run(
            [
                python,
                "scripts/check_run_complete.py",
                "--config",
                str(config_path),
                "--output",
                str(bench_output),
            ],
            env,
        )
        rows = [
            json.loads(line)
            for p in bench_output.glob("benchmark-rank*.jsonl")
            for line in p.read_text().splitlines()
        ]
        metrics = {}
        for method in config["benchmark"]["methods"]:
            selected = [r for r in rows if r["method"] == method]
            values = {
                "tokens_per_second": sum(r["output_tokens"] for r in selected)
                / sum(r["request_seconds"] for r in selected)
            }
            if method in {"relay_p", "optimized_source_reuse"}:
                accepted = [n for r in selected for n in r["accepted_draft_lengths"]]
                proposed = [n for r in selected for n in r["proposal_lengths"]]
                if (
                    not proposed
                    or len(accepted) != len(proposed)
                    or any(
                        a < 0 or a > p for a, p in zip(accepted, proposed, strict=True)
                    )
                ):
                    raise ValueError("invalid proposed-token acceptance denominator")
                values.update(
                    accepted_draft_tokens=sum(accepted),
                    proposed_draft_tokens=sum(proposed),
                    proposed_token_acceptance=sum(accepted) / sum(proposed),
                    mean_cycle_progress=sum(a + 1 for a in accepted) / len(accepted),
                )
            metrics[method] = values
        for value in metrics.values():
            value["speedup_vs_ar"] = (
                value["tokens_per_second"] / metrics["native_ar"]["tokens_per_second"]
            )
            value["native_throughput_retained"] = (
                value["tokens_per_second"]
                / metrics["native_target_dflash"]["tokens_per_second"]
            )
        (bench_output / "controlled-metrics.json").write_text(
            json.dumps(
                {
                    "checkpoint": check,
                    "methods": metrics,
                    "acceptance_scope": "All verified draft positions, before final EOS/output-cap trimming. Target-supplied positions excluded. Task scoring and paired uncertainty follow on CPU.",
                },
                indent=2,
            )
            + "\n"
        )
    (output / "experiment-complete.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "experiment": trial["name"] if trial else args.experiment,
                "evaluated_steps": saves,
                "pilot": pilot,
                "trial": trial,
                "evaluation_role": budget["role"]
                if budget
                else "historical development-exposed evaluation",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
