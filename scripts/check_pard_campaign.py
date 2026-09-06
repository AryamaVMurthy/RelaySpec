"""Audit each timed PARD request against its separate verifier-observed replica."""

import argparse
import hashlib
import json
import math
import os
import random
import struct
from pathlib import Path

from relayspec.pard_adapter import (
    PARD_SOURCE_SHA256,
    trim_pard_tokens,
    verify_pard_decisions,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    groups, inputs = {}, {}
    for rank in range(4):
        worker_path = output / f"campaign-rank{rank}.json"
        worker = json.loads(worker_path.read_text())
        if (
            worker["status"] != "pass"
            or worker["rank"] != rank
            or worker["config_sha256"] != digest(args.config)
            or worker["upstream_source_sha256"] != PARD_SOURCE_SHA256
            or not worker["frozen_parameter_versions_and_storage_unchanged"]
        ):
            raise ValueError("PARD worker lacks source/frozen-parameter verification")
        path = output / f"benchmark-rank{rank}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != config["requests"] // 4 * 2:
            raise ValueError("PARD worker lacks declared requests")
        for row in rows:
            name = row["method"]
            if (
                row["rank"] != rank
                or name not in config["methods"]
                or not math.isfinite(row["request_seconds"])
                or row["request_seconds"] <= 0
            ):
                raise ValueError("invalid PARD request")
            group = groups.setdefault(row["problem_id"], {})
            if name in group:
                raise ValueError("duplicate PARD method/request")
            group[name] = row
            raw = row["raw_output_ids"]
            tokens = trim_pard_tokens(
                raw, max_new_tokens=config["max_new_tokens"], eos_token_id=151645
            )
            if (
                tokens != row["output_ids"]
                or len(tokens) != row["output_tokens"]
                or not tokens
                or (len(tokens) < config["max_new_tokens"] and tokens[-1] != 151645)
                or row["output_hash"]
                != hashlib.sha256(struct.pack(f"<{len(tokens)}i", *tokens)).hexdigest()
            ):
                raise ValueError("PARD counted output/hash/EOS/cap differs")
            if name == "pard":
                proof = row["verification_replica"]
                if (
                    proof["token_ids"] != raw
                    or proof["accepted_lengths"] != row["acceptance_lengths"]
                    or proof["target_calls"] != row["target_calls"]
                    or proof["draft_calls"] != row["draft_calls"]
                    or not math.isfinite(proof["request_seconds"])
                    or proof["request_seconds"] <= 0
                ):
                    raise ValueError("PARD timing does not match its verified replica")
                verify_pard_decisions(
                    raw, row["acceptance_lengths"], proof["target_trace"]
                )
            elif row["verification_replica"] is not None or row["acceptance_lengths"]:
                raise ValueError("PARD AR control has speculative decisions")
            if row["accepted_draft_lengths"] != [
                n - 1 for n in row["acceptance_lengths"]
            ] or row["proposal_lengths"] != [12] * len(row["acceptance_lengths"]):
                raise ValueError("PARD acceptance summary differs")
        inputs.update({p.name: digest(p) for p in (path, worker_path)})
    records = [
        r
        for r in json.loads(Path(config["manifest_path"]).read_text())["records"]
        if r["benchmark"] == "math500"
    ]
    random.Random(config["seed"]).shuffle(records)
    if set(groups) != {r["problem_id"] for r in records[: config["requests"]]} or any(
        set(g) != set(config["methods"])
        or g["pard"]["input_ids"] != g["native_ar"]["input_ids"]
        for g in groups.values()
    ):
        raise ValueError("PARD lacks the fixed paired development requests")
    result = {
        "status": "pass",
        "config_sha256": digest(args.config),
        "records": len(groups) * 2,
        "requests": len(groups),
        "methods": config["methods"],
        "input_sha256": inputs,
        "ar_exact_count": sum(
            g["pard"]["output_ids"] == g["native_ar"]["output_ids"]
            for g in groups.values()
        ),
        "verification_replica_seconds": sum(
            g["pard"]["verification_replica"]["request_seconds"]
            for g in groups.values()
        ),
        "scope": config["scope"],
    }
    (output / "completion-gate.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
