"""Gate small drafter updates through fitting, reload, and actual decoding."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch
import yaml

from relayspec.mapper_campaign import campaign_references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    fits = Path(os.environ["RELAYSPEC_FIT_OUTPUT"])
    python = os.environ["RELAYSPEC_PYTHON"]
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    declared_config = yaml.safe_load(args.config.read_text())
    declared_settings = declared_config["adaptation_pilot"]
    family = declared_config["proposer"]["family"]
    prerequisite = declared_settings.get("prerequisite_gate")
    if prerequisite:
        prerequisite_path = Path(prerequisite["path"])
        prior = json.loads(prerequisite_path.read_text())
        if (
            hashlib.sha256(prerequisite_path.read_bytes()).hexdigest()
            != prerequisite["sha256"]
            or prior.get("status") != "pass"
            or prior.get("duplicate_seed_weights_and_decoding_bit_identical")
            is not True
            or prior.get("zero_update_decoding_bit_identical") is not True
            or len(prior.get("fitting", [])) != 4
            or any(
                fit["feature_cache_index_sha256"]
                != declared_settings["feature_cache_index_sha256"]
                or fit["initial_mapper_sha256"]
                != declared_settings["initial_mapper_sha256"]
                for fit in prior["fitting"]
            )
        ):
            raise ValueError(
                "adaptation calibration requires its exact cache/mapper compatibility pilot"
            )
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
    for rank in range(4):
        folder = fits / f"rank{rank}"
        gate = json.loads((folder / "adaptation-fit-gate.json").read_text())
        if (
            gate["status"] != "pass"
            or not gate["inherited_weights_unchanged"]
            or gate["updates"] != declared_settings["pilot_updates"]
            or gate["prepared_distinct_examples"]
            != declared_settings["pilot_distinct_examples"]
        ):
            raise ValueError("incomplete bounded adaptation fit")
        if family == "eagle3":
            checks = gate.get("eagle_initial_cache_equivalence", [])
            if len(checks) != 4 or any(
                c.get("status") != "pass" or c.get("logits_bit_identical") is not True
                for c in checks
            ):
                raise ValueError(
                    "EAGLE adaptation lacks its shifted-prefix cache proof"
                )
        gates.append(gate)
        destination = output / "fitting" / f"rank{rank}"
        destination.mkdir(parents=True, exist_ok=False)
        for path in folder.glob("*.json*"):
            shutil.copy2(path, destination / path.name)
    left, right = [
        torch.load(
            fits / f"rank{rank}/adaptation.pt", weights_only=True, map_location="cpu"
        )
        for rank in (2, 3)
    ]
    if left.keys() != right.keys() or left["weights"].keys() != right["weights"].keys():
        raise ValueError("duplicate adaptation payload structures differ")
    if any(
        not torch.equal(value, right["weights"][name])
        for name, value in left["weights"].items()
    ):
        raise ValueError("duplicate-seed drafter updates are not bit identical")
    config = yaml.safe_load(args.config.read_text())
    settings = config.pop("adaptation_pilot")
    variants = {
        "relay_initial": settings["initial_mapper"],
        "relay_connector_ce": str(fits / "rank0/mapper.pt"),
        "relay_lora8": settings["initial_mapper"],
        "relay_lora32": settings["initial_mapper"],
        "relay_lora32_duplicate": settings["initial_mapper"],
        "relay_lora_zero": settings["initial_mapper"],
    }
    config["relay_probe"]["variants"] = variants
    config["relay_probe"]["drafter_updates"] = {
        "relay_lora8": str(fits / "rank1/adaptation.pt"),
        "relay_lora32": str(fits / "rank2/adaptation.pt"),
        "relay_lora32_duplicate": str(fits / "rank3/adaptation.pt"),
        "relay_lora_zero": str(fits / "rank2/zero-adaptation.pt"),
    }
    config["benchmark"]["methods"] = [
        *campaign_references(family, config["benchmark"]["methods"]),
        *variants,
    ]
    config["benchmark"]["max_prompts"] = 8
    config["generation"]["max_new_tokens"] = 128
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
    duplicate = [
        {(r["problem_id"], r["repetition"]): r for r in rows if r["method"] == name}
        for name in ("relay_lora32", "relay_lora32_duplicate")
    ]
    if len(duplicate[0]) != 8 or duplicate[0].keys() != duplicate[1].keys():
        raise ValueError("duplicate adaptation decoding requests differ")
    for key, row in duplicate[0].items():
        if any(
            row[field] != duplicate[1][key][field]
            for field in ("output_hash", "output_tokens")
            + (
                ("accepted_draft_lengths", "proposal_lengths")
                if family == "dflash"
                else ("acceptance_lengths", "target_calls", "draft_calls")
            )
        ):
            raise ValueError("duplicate adapted drafter decoding differs")
    (output / "adaptation-pilot-gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "fitting": gates,
                "zero_update_decoding_bit_identical": True,
                "duplicate_seed_weights_and_decoding_bit_identical": True,
                "total_seconds": time.perf_counter() - started,
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "scope": f"{family} connector-only CE versus frozen-connector drafter LoRA resource/correctness pilot. Full matched-compute/data budgets and scientific comparisons remain unexecuted.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
