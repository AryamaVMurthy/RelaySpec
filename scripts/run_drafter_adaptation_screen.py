"""Run a declared four-worker adaptation screen after the strict pilot passes."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    settings = config["adaptation_pilot"]
    prerequisite = settings["prerequisite_gate"]
    prerequisite_path = Path(prerequisite["path"])
    pilot = json.loads(prerequisite_path.read_text())
    if (
        digest(prerequisite_path) != prerequisite["sha256"]
        or pilot["status"] != "pass"
        or not pilot["duplicate_seed_weights_and_decoding_bit_identical"]
    ):
        raise ValueError("screen requires the source-verified strict adaptation pilot")
    trials = settings["worker_trials"]
    if (
        len(trials) != 4
        or settings["pilot_distinct_examples"] != 512
        or settings["pilot_updates"] != 128
    ):
        raise ValueError("screen differs from its equal-data calibration contract")
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    fits = Path(os.environ["RELAYSPEC_FIT_OUTPUT"])
    python = os.environ["RELAYSPEC_PYTHON"]
    variants = {"relay_initial": settings["initial_mapper"]}
    updates = {}
    provenance = {"relay_initial": {"mapper_sha256": settings["initial_mapper_sha256"]}}
    for name, reused in settings["reused_variants"].items():
        if digest(Path(reused["mapper"])) != reused["mapper_sha256"]:
            raise ValueError("reused mapper checkpoint changed")
        variants[name] = reused["mapper"]
        if "drafter_update" in reused:
            if (
                digest(Path(reused["drafter_update"]))
                != reused["drafter_update_sha256"]
            ):
                raise ValueError("reused drafter checkpoint changed")
            updates[name] = reused["drafter_update"]
        provenance[name] = reused
    started = time.perf_counter()
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
    zero_rank = None
    for rank, trial in enumerate(trials):
        folder = fits / f"rank{rank}"
        gate = json.loads((folder / "adaptation-fit-gate.json").read_text())
        if (
            gate["status"] != "pass"
            or gate["worker_trial"] != trial
            or gate["config_sha256"] != digest(args.config)
            or gate["updates"] != 128
            or gate["distinct_examples"] != 512
            or not gate["inherited_weights_unchanged"]
            or gate["initial_mapper_sha256"] != settings["initial_mapper_sha256"]
        ):
            raise ValueError("screen fit differs from its declared worker")
        if rank and gate["ordered_record_files"] != gates[0]["ordered_record_files"]:
            raise ValueError("screen workers used different data")
        name = "relay_" + trial["name"]
        if trial["lora_rank"]:
            path = folder / "adaptation.pt"
            variants[name] = settings["initial_mapper"]
            updates[name] = str(path)
            provenance[name] = {
                "mapper_sha256": settings["initial_mapper_sha256"],
                "drafter_update_sha256": gate["checkpoint_sha256"],
            }
            if (
                gate["export"]["status"] != "pass"
                or not gate["export"]["merged_reload_bit_identical"]
            ):
                raise ValueError("screen LoRA export did not pass")
            zero_rank = rank if zero_rank is None else zero_rank
        else:
            path = folder / "mapper.pt"
            variants[name] = str(path)
            provenance[name] = {"mapper_sha256": gate["checkpoint_sha256"]}
        if digest(path) != gate["checkpoint_sha256"]:
            raise ValueError("screen checkpoint differs from its fitting gate")
        destination = output / "fitting" / f"rank{rank}"
        destination.mkdir(parents=True, exist_ok=False)
        for path in folder.glob("*.json*"):
            shutil.copy2(path, destination / path.name)
        gates.append(gate)
    if zero_rank is None:
        raise ValueError("screen requires a zero-LoRA decoding identity control")
    zero = fits / f"rank{zero_rank}" / "zero-adaptation.pt"
    variants["relay_lora_zero"] = settings["initial_mapper"]
    updates["relay_lora_zero"] = str(zero)
    provenance["relay_lora_zero"] = {
        "mapper_sha256": settings["initial_mapper_sha256"],
        "drafter_update_sha256": digest(zero),
    }
    del config["adaptation_pilot"]
    config["relay_probe"]["variants"] = variants
    config["relay_probe"]["drafter_updates"] = updates
    config["benchmark"]["methods"] = [
        "native_ar",
        "native_target_dflash",
        "optimized_source_reuse",
        *variants,
    ]
    path = output / "campaign-config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "screen-provenance.json").write_text(
        json.dumps(
            {
                "config_sha256": digest(args.config),
                "campaign_sha256": digest(path),
                "prerequisite_gate_sha256": prerequisite["sha256"],
                "variants": provenance,
                "scope": settings["scope"],
            },
            indent=2,
        )
        + "\n"
    )
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
    (output / "adaptation-screen-gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "fitting": gates,
                "config_sha256": digest(args.config),
                "provenance_sha256": digest(output / "screen-provenance.json"),
                "zero_update_decoding_bit_identical": True,
                "total_seconds": time.perf_counter() - started,
                "scope": settings["scope"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
