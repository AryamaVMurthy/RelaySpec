"""Report passed fitting invariants while preserving the failed exact-AR gate."""

import argparse
import hashlib
import json
from pathlib import Path

from relayspec.sd_square_adapter import committed_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs, fits, rows, compact = {}, [], [], []
    config_path = Path("configs/submission/baselines/sd-square-pilot.json")
    config = json.loads(config_path.read_text())
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    inputs[str(config_path)] = config_sha
    setup_config_path = Path(config["setup_config"])
    setup_path = args.run / "setup-gate.json"
    setup = json.loads(setup_path.read_text())
    setup_config = json.loads(setup_config_path.read_text())
    ledger_path = Path("reports/mapper-scaling-20260905/sd-square-pilot/jobs.json")
    source_path = args.run / "source-commit.txt"
    if (
        hashlib.sha256(setup_config_path.read_bytes()).hexdigest()
        != config["setup_config_sha256"]
        or setup["status"] != "pass"
        or setup["config_sha256"] != config["setup_config_sha256"]
        or setup["source_sha256"] != setup_config["source_sha256"]
        or any(
            setup["versions"][name] != version
            for name, version in setup_config["runtime"].items()
        )
        or source_path.read_text().strip()
        != json.loads(ledger_path.read_text())["source_commit"]
    ):
        raise ValueError("SD-square source or environment provenance differs")
    for path in (setup_config_path, setup_path, ledger_path, source_path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for rank in range(4):
        documents = {}
        for name in ("pilot", "training", "decoding"):
            path = args.run / f"{name}-rank{rank}.json"
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            documents[name] = json.loads(path.read_text())
        fit, training, row = [documents[k] for k in ("pilot", "training", "decoding")]
        if (
            fit["phase"] != "complete"
            or fit.get("error")
            or fit["rank"] != rank
            or fit["config_sha256"] != config_sha
            or fit["setup_gate_sha256"] != inputs[str(setup_path)]
            or fit["training"] != training
            or [r["step"] for r in training] != list(range(1, config["updates"] + 1))
            or fit["distinct_records_seen"] != len({r["record"] for r in training})
            or len(set(fit["ordered_pool_files"])) != 512
            or [r["record"] for r in training] != fit["ordered_pool_files"][:4]
            or fit["objective"] != config["worker_objectives"][rank]
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
            raise ValueError("SD-square fitting evidence is incomplete")
        counted, raw = committed_tokens(row["trace"], config["max_new_tokens"], 151645)
        if (
            counted != row["output_ids"]
            or raw != row["raw_committed_ids"]
            or any(b["tokens"] != b["verifier_argmax"] for b in row["trace"])
        ):
            raise ValueError("SD-square committed-token observation differs")
        exact = row["output_ids"] == row["ar_ids"]
        if (
            fit["ar_exact"] != exact
            or row["ar_exact"] != exact
            or fit["status"] != ("pass" if exact else "fail")
        ):
            raise ValueError("SD-square AR identity result was changed")
        first = next(
            (
                i
                for i, (a, b) in enumerate(zip(row["output_ids"], row["ar_ids"]))
                if a != b
            ),
            None,
        )
        compact.append(
            {
                k: fit[k]
                for k in (
                    "rank",
                    "objective",
                    "trainable_parameters",
                    "trainable_sha256",
                    "frozen_drafter_sha256",
                    "training_seconds",
                    "model_load_seconds",
                    "total_seconds",
                    "peak_gpu_bytes",
                    "checkpoint",
                    "checkpoint_sha256",
                    "attention",
                    "storage_dtypes",
                )
            }
            | {"ar_exact": exact, "first_difference": first, "training": training}
        )
        fits.append(fit)
        rows.append(row)
    for a, b in ((0, 1), (2, 3)):
        if any(
            fits[a][k] != fits[b][k]
            for k in (
                "trainable_sha256",
                "frozen_drafter_sha256",
                "seed",
                "ordered_pool_files",
            )
        ):
            raise ValueError("SD-square duplicate fitting differs")
        if [(r["loss"], r["gradient_norm"]) for r in fits[a]["training"]] != [
            (r["loss"], r["gradient_norm"]) for r in fits[b]["training"]
        ]:
            raise ValueError("SD-square duplicate losses or gradients differ")
        if any(
            rows[a][k] != rows[b][k]
            for k in ("input_ids", "output_ids", "ar_ids", "trace")
        ):
            raise ValueError("SD-square duplicate decoding differs")
    args.output.write_text(
        json.dumps(
            {
                "status": "training_and_reload_passed_ar_identity_failed"
                if not all(r["ar_exact"] for r in rows)
                else "pilot_passed",
                "input_sha256": inputs,
                "records": compact,
                "scope": config["scope"]
                + " Target frozen-parameter version/storage invariants and exact "
                "drafter hashes are separate checks. Exact-AR failure remains a failed gate. "
                "No full512-example baseline result, numerical-cause attribution or task-quality conclusion.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
