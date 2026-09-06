"""Declare paired decoding of all four audited small-pool epoch endpoints."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from build_sd_square_campaign import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    registry_path = Path("reports/external-baselines-20260906/sd-square-epochs.json")
    registry = json.loads(registry_path.read_text())
    fit_config = Path("configs/submission/baselines/sd-square-epochs.json")
    ledger_path = Path("reports/mapper-scaling-20260905/sd-square-epochs/jobs.json")
    ledger = json.loads(ledger_path.read_text())
    if (
        registry["status"] != "complete"
        or (args.run / "source-commit.txt").read_text().strip()
        != ledger["source_commit"]
    ):
        raise ValueError("SD-square epoch fitting is incomplete or uses another source")
    evidence = {}
    for original, sha in registry["input_sha256"].items():
        path = Path(original)
        if path.is_absolute():
            path = args.run / path.name
        if digest(path) != sha:
            raise ValueError(f"SD-square epoch input changed: {path}")
        evidence[path.name] = sha
    gate_path = args.run / "sd-square-full-pilot-gate.json"
    gate = json.loads(gate_path.read_text())
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        for kind in ("pilot", "training", "decoding"):
            for rank in range(4):
                path = args.run / f"{kind}-rank{rank}.json"
                (folder / path.name).symlink_to(path.resolve())
        env = os.environ.copy()
        env["RELAYSPEC_OUTPUT"] = str(folder)
        subprocess.run(
            [
                sys.executable,
                "scripts/summarize_sd_square_fullpilot.py",
                "--config",
                str(fit_config),
            ],
            env=env,
            check=True,
            capture_output=True,
        )
        if json.loads((folder / gate_path.name).read_text()) != gate:
            raise ValueError("SD-square epoch gate does not reproduce")
    variants = {"native_ar": {"kind": "ar"}, "sd2_independent": {"kind": "independent"}}
    for fit, record in zip(gate["fitting"], registry["records"], strict=True):
        fields = (
            "checkpoint",
            "checkpoint_sha256",
            "trainable_sha256",
            "updates",
            "epochs",
            "learning_rate",
            "objective",
        )
        if (
            any(fit[k] != record[k] for k in fields)
            or fit["epochs"] != 2 ** fit["rank"]
        ):
            raise ValueError("SD-square epoch descriptor differs from raw fitting")
        if (
            args.check_files
            and digest(Path(fit["checkpoint"])) != fit["checkpoint_sha256"]
        ):
            raise ValueError("SD-square epoch checkpoint file changed")
        variants[f"sd2_epoch{fit['epochs']}"] = {
            "kind": "steering",
            **{k: fit[k] for k in fields},
            "distinct_examples": 512,
        }
    source_path = Path("configs/submission/baselines/sd-square-compatibility.json")
    protocol_path = Path(
        "configs/submission/baselines/external-verification-protocol.json"
    )
    result = {
        "source_config": str(source_path),
        "source_config_sha256": digest(source_path),
        "verification_protocol": str(protocol_path),
        "verification_protocol_sha256": digest(protocol_path),
        "seed": 1729,
        "manifest_path": "configs/eval_manifest.json",
        "requests": 8,
        "request_offset": 8 * args.shard_index,
        "total_development_requests": 16,
        "max_new_tokens": 256,
        "observer_pilot_tokens": 64,
        "warmup_tokens": 16,
        "inference_precision": {
            "target": "float16",
            "drafter": "bfloat16",
            "steering": "bfloat16",
            "autocast": "bfloat16",
        },
        "variants": variants,
        "fitting_input_sha256": {**evidence, "epoch_registry": digest(registry_path)},
        "original_exact_ar_gates": registry["original_exact_ar_gates"],
        "scope": "Four SD-square epoch endpoints fitted on the same512 examples with frozen KL4e-6 rate. "
        "Separate1/2/4/8-epoch decay schedules, with one-epoch exact fitting reproduction. "
        "Sixteen paired exposed development prompts in two disjoint eight-request shards,256-token cap. "
        "Public FP16 target and BF16 drafter/steering storage with BF16 autocast. Training and inference "
        "fingerprints are checked separately. Immediate/deferred observation must preserve all decisions. "
        "Full request timing includes GPU observation and excludes CPU materialization and detokenization. "
        "Correct runtime-local AR and independent-drafter controls. No full-answer quality, untouched "
        "confirmation, isolated serving-memory or cross-runtime algorithm-only ranking. Prior AR failures remain preserved.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
