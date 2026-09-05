"""Prove uninterrupted versus resumed adaptation before budget continuations."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch
import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(exact(value, right[key]) for key, value in left.items())
        )
    if isinstance(left, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(exact(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    settings = config["adaptation_pilot"]
    prerequisite = settings["prerequisite_gate"]
    source = Path(prerequisite["path"])
    if (
        digest(source) != prerequisite["sha256"]
        or json.loads(source.read_text())["status"] != "pass"
    ):
        raise ValueError("continuation requires the verified parent fitting screen")
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    fits = Path(os.environ["RELAYSPEC_FIT_OUTPUT"])
    python = os.environ["RELAYSPEC_PYTHON"]
    subprocess.run(
        [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            "scripts/fit_drafter_adaptation_pilot.py",
            "--config",
            str(args.config),
        ],
        check=True,
    )
    gates = []
    for rank, trial in enumerate(settings["worker_trials"]):
        folder = fits / f"rank{rank}"
        gate = json.loads((folder / "adaptation-fit-gate.json").read_text())
        if (
            gate["status"] != "pass"
            or gate["worker_trial"] != trial
            or gate["updates"] != 256
            or gate["start_step"] != (128 if rank % 2 else 0)
            or gate["distinct_examples"] != 512
            or not gate["inherited_weights_unchanged"]
            or digest(folder / "training-state.pt") != gate["training_state_sha256"]
        ):
            raise ValueError("continuation worker did not follow its declaration")
        destination = output / "fitting" / f"rank{rank}"
        destination.mkdir(parents=True, exist_ok=False)
        for path in folder.glob("*.json*"):
            shutil.copy2(path, destination / path.name)
        gates.append(gate)
    pairs = []
    for left_rank, right_rank in [(0, 1), (2, 3)]:
        left, right = [
            torch.load(
                fits / f"rank{r}/training-state.pt",
                weights_only=True,
                map_location="cpu",
            )
            for r in (left_rank, right_rank)
        ]
        if not all(
            exact(left[k], right[k])
            for k in ("trainable", "optimizer", "ordered_record_files")
        ):
            raise ValueError("resumed trainable weights, Adam state, or data differ")
        if left_rank == 2:
            a, b = [
                torch.load(
                    fits / f"rank{r}/adaptation.pt",
                    weights_only=True,
                    map_location="cpu",
                )
                for r in (left_rank, right_rank)
            ]
            if not exact(a, b):
                raise ValueError("resumed merged drafter update differs")
        pairs.append(
            {
                "workers": [left_rank, right_rank],
                "weights_and_optimizer_bit_identical": True,
            }
        )
    del config["adaptation_pilot"]
    initial = settings["initial_mapper"]
    config["relay_probe"]["variants"] = {
        "relay_initial": initial,
        "relay_ce_full": str(fits / "rank0/mapper.pt"),
        "relay_ce_resumed": str(fits / "rank1/mapper.pt"),
        "relay_lora_full": initial,
        "relay_lora_resumed": initial,
        "relay_lora_zero": initial,
    }
    config["relay_probe"]["drafter_updates"] = {
        "relay_lora_full": str(fits / "rank2/adaptation.pt"),
        "relay_lora_resumed": str(fits / "rank3/adaptation.pt"),
        "relay_lora_zero": str(fits / "rank2/zero-adaptation.pt"),
    }
    config["benchmark"]["methods"] = [
        "native_ar",
        "native_target_dflash",
        "optimized_source_reuse",
        *config["relay_probe"]["variants"],
    ]
    path = output / "campaign-config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    subprocess.run(
        [
            python,
            "scripts/run_mapper_campaign.py",
            "--config",
            str(path),
            "--equal-methods",
            "relay_initial",
            "relay_lora_zero",
        ],
        env={**os.environ, "RELAYSPEC_OUTPUT": str(output / "evaluation")},
        check=True,
    )
    rows = [
        json.loads(line)
        for path in (output / "evaluation").glob("benchmark-rank*.jsonl")
        for line in path.read_text().splitlines()
    ]
    for family in ("ce", "lora"):
        pair = [
            {
                (r["problem_id"], r["repetition"]): r
                for r in rows
                if r["method"] == f"relay_{family}_{suffix}"
            }
            for suffix in ("full", "resumed")
        ]
        if (
            len(pair[0]) != 8
            or pair[0].keys() != pair[1].keys()
            or any(
                any(
                    row[k] != pair[1][key][k]
                    for k in (
                        "output_hash",
                        "output_tokens",
                        "accepted_draft_lengths",
                        "proposal_lengths",
                    )
                )
                for key, row in pair[0].items()
            )
        ):
            raise ValueError("resumed actual decoding differs")
    (output / "adaptation-continuation-gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "pairs": pairs,
                "fitting": gates,
                "config_sha256": digest(args.config),
                "decoding_bit_identical": True,
                "scope": settings["scope"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
