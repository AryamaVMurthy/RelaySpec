"""Verify prospective full-pool training without rewriting old exact-AR failures."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from relayspec.sd_square_adapter import committed_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    protocol = config["verification_protocol"]
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    fits, rows, inputs = [], [], {}
    for rank in range(4):
        docs = {}
        for name in ("pilot", "training", "decoding"):
            path = output / f"{name}-rank{rank}.json"
            docs[name] = json.loads(path.read_text())
            inputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        fit, training, row = [docs[n] for n in ("pilot", "training", "decoding")]
        seen = [name for step in training for name in step["records"]]
        if (
            fit["status"] != "pass"
            or fit["phase"] != "complete"
            or fit["rank"] != rank
            or fit["config_sha256"]
            != hashlib.sha256(args.config.read_bytes()).hexdigest()
            or fit["verification_protocol_sha256"] != protocol["sha256"]
            or fit["causal_gate_sha256"] != protocol["causal_gate_sha256"]
            or fit["training"] != training
            or [r["step"] for r in training] != list(range(1, 129))
            or len(seen) != 512
            or len(set(seen)) != 512
            or seen != fit["ordered_pool_files"]
            or fit["batch_size"] != 4
            or fit["distinct_records_seen"] != 512
            or any(
                len(r["records"]) != 4
                or len(r["lengths"]) != 4
                or not 0 < r["loss_tokens"] <= sum(r["lengths"])
                for r in training
            )
            or not all(
                fit[k]
                for k in (
                    "zero_guidance_bit_identical",
                    "steering_reload_bit_identical",
                    "optimizer_reload_bit_identical",
                    "frozen_parameter_versions_and_storage_unchanged",
                )
            )
        ):
            raise ValueError("full-pool SD-square fitting invariant failed")
        if fit["original_exact_ar_gates"] != {
            "sd_square_job27869": "failed_preserved",
            "pard_job27831": "failed_preserved",
        }:
            raise ValueError("prospective pilot rewrote earlier AR failures")
        counted, raw = committed_tokens(row["trace"], config["max_new_tokens"], 151645)
        if (
            counted != row["output_ids"]
            or raw != row["raw_committed_ids"]
            or not row["verifier_token_checks_passed"]
            or any(b["tokens"] != b["verifier_argmax"] for b in row["trace"])
            or row["ar_exact"] != (row["output_ids"] == row["ar_ids"])
            or fit["ar_exact"] != row["ar_exact"]
        ):
            raise ValueError("prospective SD-square token verification differs")
        fits.append(fit)
        rows.append(row)
        if "followup_screen" in config and (
            fit["learning_rate"] != config["worker_learning_rates"][rank]
            or fit["learning_rate_end"] != config["worker_learning_rates"][rank] / 10
            or fit["parent_full_pilot_gate_sha256"]
            != config["followup_screen"]["pilot_gate_sha256"]
        ):
            raise ValueError("SD-square rate screen changed its declared trial")
    for a, b in ((0, 1), (2, 3)):
        if "followup_screen" in config:
            continue  # Distinct declared rates, using the prior exact-replica pilot.
        if any(
            fits[a][k] != fits[b][k]
            for k in (
                "trainable_sha256",
                "frozen_drafter_sha256",
                "objective",
                "ordered_pool_files",
                "seed",
            )
        ):
            raise ValueError("full-pool duplicate fit differs")
        if [
            {k: v for k, v in r.items() if k != "seconds"} for r in fits[a]["training"]
        ] != [
            {k: v for k, v in r.items() if k != "seconds"} for r in fits[b]["training"]
        ]:
            raise ValueError("full-pool duplicate losses or gradients differ")
        if any(
            rows[a][k] != rows[b][k]
            for k in ("input_ids", "output_ids", "ar_ids", "trace")
        ):
            raise ValueError("full-pool duplicate decoding differs")
    (output / "sd-square-full-pilot-gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "verification_protocol_sha256": protocol["sha256"],
                "input_sha256": inputs,
                "fitting": fits,
                "ar_exact_workers": [r["ar_exact"] for r in rows],
                "original_exact_ar_gates": fits[0]["original_exact_ar_gates"],
                "scope": config["scope"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
