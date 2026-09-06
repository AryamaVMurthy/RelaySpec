"""Declare common decoding only after both complete-pool fitting batches pass."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--shard-index", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    variants = {
        "native_ar": {"kind": "ar"},
        "sd2_independent": {"kind": "independent"},
        "sd2_zero_guidance": {"kind": "zero_guidance"},
    }
    evidence, gates = {}, []
    for run, study, config_name in zip(
        args.runs,
        ("sd-square-fullpilot", "sd-square-rates"),
        ("sd-square-full-pilot", "sd-square-rate-screen"),
        strict=True,
    ):
        config_path = Path(f"configs/submission/baselines/{config_name}.json")
        config = json.loads(config_path.read_text())
        gate_path = run / "sd-square-full-pilot-gate.json"
        gate = json.loads(gate_path.read_text())
        ledger = Path(f"reports/mapper-scaling-20260905/{study}/jobs.json")
        if (run / "source-commit.txt").read_text().strip() != json.loads(
            ledger.read_text()
        )["source_commit"]:
            raise ValueError("SD-square fitting source differs from recorded job")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for prefix in ("pilot", "training", "decoding"):
                for rank in range(4):
                    path = run / f"{prefix}-rank{rank}.json"
                    (directory / path.name).symlink_to(path.resolve())
                    evidence[f"{study}/{path.name}"] = digest(path)
            env = os.environ.copy()
            env["RELAYSPEC_OUTPUT"] = str(directory)
            subprocess.run(
                [
                    sys.executable,
                    "scripts/summarize_sd_square_fullpilot.py",
                    "--config",
                    str(config_path),
                ],
                env=env,
                check=True,
                capture_output=True,
            )
            if json.loads((directory / gate_path.name).read_text()) != gate:
                raise ValueError("SD-square full fitting evidence does not reproduce")
        evidence.update(
            {
                f"{study}/gate": digest(gate_path),
                f"{study}/config": digest(config_path),
                f"{study}/source": digest(run / "source-commit.txt"),
                f"{study}/ledger": digest(ledger),
            }
        )
        gates.append((gate, digest(gate_path)))
        ranks = [0, 2] if study == "sd-square-fullpilot" else range(4)
        for rank in ranks:
            fit = gate["fitting"][rank]
            lr = config.get("worker_learning_rates", [config["learning_rate"]] * 4)[
                rank
            ]
            name = f"sd2_{fit['objective']}_lr{lr:g}"
            variants[name] = {
                "kind": "steering",
                "objective": fit["objective"],
                "learning_rate": lr,
                "checkpoint": fit["checkpoint"],
                "checkpoint_sha256": fit["checkpoint_sha256"],
                "trainable_sha256": fit["trainable_sha256"],
                "fitting_gate_sha256": digest(gate_path),
                "fitting_rank": rank,
                "distinct_examples": 512,
                "updates": 128,
            }
    if len(variants) != 9 or any(
        f["parent_full_pilot_gate_sha256"] != gates[0][1]
        for f in gates[1][0]["fitting"]
    ):
        raise ValueError("SD-square campaign is missing a rate or uses another pilot")
    if args.check_files:
        for value in variants.values():
            if (
                value["kind"] == "steering"
                and digest(Path(value["checkpoint"])) != value["checkpoint_sha256"]
            ):
                raise ValueError("SD-square campaign checkpoint changed")
    protocol_path = Path(
        "configs/submission/baselines/external-verification-protocol.json"
    )
    protocol = json.loads(protocol_path.read_text())
    source_config = Path("configs/submission/baselines/sd-square-compatibility.json")
    result = {
        "source_config": str(source_config),
        "source_config_sha256": digest(source_config),
        "verification_protocol": str(protocol_path),
        "verification_protocol_sha256": digest(protocol_path),
        "seed": 1729,
        "inference_precision": {
            "target": "float16",
            "drafter": "bfloat16",
            "steering": "bfloat16",
            "autocast": "bfloat16",
        },
        "manifest_path": "configs/eval_manifest.json",
        "requests": 8,
        "request_offset": 8 * args.shard_index,
        "total_development_requests": 16,
        "max_new_tokens": 256,
        "observer_pilot_tokens": 64,
        "warmup_tokens": 16,
        "variants": variants,
        "fitting_input_sha256": evidence,
        "original_exact_ar_gates": protocol["original_exact_ar_gates"],
        "scope": "Public eval.py inference storage uses FP16 target and BF16 drafter/steering with BF16 autocast. "
        "Training and converted inference steering fingerprints are separately verified. "
        "Six predeclared512-example SD-square objective/rate fits, zero-guidance and "
        "independent-drafter controls, and correct runtime-local AR. Sixteen paired exposed "
        "development requests with256-token cap. Immediate/deferred observer equality is "
        "checked before measurements. GPU decision capture is timed, CPU materialization and "
        "Each ten-minute job covers one disjoint eight-request shard of the fixed sixteen. "
        "verification occur after the timer. Shared resident steering storage prevents isolated "
        "serving-memory claims. No full-answer quality, untouched confirmation or cross-runtime "
        "algorithm-only ranking. Old exact-AR failures are preserved.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
