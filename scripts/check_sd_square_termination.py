"""Require reproduction of the public physical-slot guard on the failed warmup."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

from relayspec.sd_square_adapter import termination_status


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    manifest = [
        r
        for r in json.loads(Path(config["manifest_path"]).read_text())["records"]
        if r["benchmark"] == "math500"
    ]
    random.Random(config["seed"]).shuffle(manifest)
    records = manifest[
        config["request_offset"] : config["request_offset"] + config["requests"]
    ]
    expected = config["variants"]["sd2_selected"]
    probes, inputs = [], {}
    for rank, record in enumerate(records):
        path = output / f"termination-rank{rank}.json"
        row = json.loads(path.read_text())
        if (
            row["status"] != "complete"
            or row["rank"] != rank
            or row["config_sha256"] != digest(args.config)
            or not row["frozen_parameter_versions_and_storage_unchanged"]
            or row["steering_fingerprints"]
            != {
                "sd2_selected": {
                    "training_sha256": expected["trainable_sha256"],
                    "inference_sha256": expected["inference_sha256"],
                }
            }
        ):
            raise ValueError("termination worker provenance differs")
        if len(row["probes"]) != 2 or {p["requested_cap"] for p in row["probes"]} != {
            16,
            64,
        }:
            raise ValueError("termination probe coverage differs")
        for probe in row["probes"]:
            if (
                probe["problem_id"] != record["problem_id"]
                or probe["method"] != "sd2_selected"
                or not probe["observer_exact"]
                or probe["trace"] != probe["deferred_trace"]
            ):
                raise ValueError(
                    "termination probe identity or observer equality differs"
                )
            for block in probe["trace"]:
                if (
                    not 0 <= block["accepted_draft_tokens"] <= 8
                    or len(block["tokens"]) != block["accepted_draft_tokens"] + 1
                    or block["tokens"] != block["verifier_argmax"]
                ):
                    raise ValueError("termination probe differs from actual verifier")
            status = termination_status(
                probe["trace"], probe["requested_cap"], 151645, 8
            )
            if (
                any(probe[k] != v for k, v in status.items())
                or status["reason"] == "unexplained_early_stop"
            ):
                raise ValueError("public termination explanation does not reproduce")
            probes.append({k: probe[k] for k in ["problem_id", *status]})
        inputs[path.name] = digest(path)
    reproduced = [
        p
        for p in probes
        if p["problem_id"] == config["termination_diagnostic"]["failed_problem_id"]
        and p["requested_cap"] == 16
    ]
    if len(reproduced) != 1 or reproduced[0]["reason"] != "public_physical_slot_guard":
        raise ValueError("failed warmup was not reproduced")
    (output / "sd-square-numerical-gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "config_sha256": digest(args.config),
                "raw_sha256": inputs,
                "probes": probes,
                "scope": "Reproduced public physical-slot warmup termination. No full-quality campaign pass or modification of public generation statements.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
