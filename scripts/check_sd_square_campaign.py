"""Verify paired outputs, loaded steering identities and observer equivalence."""

import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path

from relayspec.sd_square_adapter import committed_tokens, termination_status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    groups, inputs, records = {}, {}, []
    fingerprints = None
    for rank in range(4):
        worker_path = output / f"campaign-rank{rank}.json"
        worker = json.loads(worker_path.read_text())
        if (
            worker["status"] != "pass"
            or worker["rank"] != rank
            or worker["config_sha256"] != config_sha
            or not worker["frozen_parameter_versions_and_storage_unchanged"]
            or worker["inference_precision"] != config["inference_precision"]
            or set(worker["observer_equality"])
            != set(config["variants"]) - {"native_ar"}
        ):
            raise ValueError("SD-square campaign worker is incomplete")
        identities = worker["steering_fingerprints"]
        expected_names = {
            n
            for n, v in config["variants"].items()
            if v["kind"] in {"steering", "zero_guidance"}
        }
        if set(identities) != expected_names or (
            fingerprints is not None and identities != fingerprints
        ):
            raise ValueError("SD-square inference identities differ across workers")
        fingerprints = identities
        for name, identity in identities.items():
            if (
                identity["training_sha256"]
                != config["variants"][name].get("trainable_sha256")
                or len(identity["inference_sha256"]) != 64
            ):
                raise ValueError(
                    "SD-square inference identity lacks training provenance"
                )
        for probe in worker["observer_equality"].values():
            if (
                probe["status"] != "pass"
                or probe["tokens"] != probe["deferred_tokens"]
                or probe["trace"] != probe["deferred_trace"]
            ):
                raise ValueError("SD-square observer equality lacks matching evidence")
        path = output / f"benchmark-rank{rank}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != config["requests"] // 4 * len(config["variants"]):
            raise ValueError("SD-square worker is missing requests")
        for row in rows:
            if (
                row["rank"] != rank
                or row["method"] not in config["variants"]
                or not math.isfinite(row["request_seconds"])
                or row["request_seconds"] <= 0
            ):
                raise ValueError("invalid SD-square campaign row")
            value = config["variants"][row["method"]]
            if (
                value.get("inference_sha256", row["steering_inference_sha256"])
                != row["steering_inference_sha256"]
            ):
                raise ValueError(
                    "SD-square row differs from selected inference fingerprint"
                )
            key = (row["problem_id"], row["repetition"])
            group = groups.setdefault(key, {})
            if row["method"] in group:
                raise ValueError("duplicate SD-square request/method")
            group[row["method"]] = row
            if value["kind"] != "ar":
                if config.get("warmup_policy"):
                    warmup = row["warmup"]
                    status = termination_status(
                        warmup["trace"], config["warmup_tokens"], 151645, 8
                    )
                    if (
                        config["warmup_policy"]
                        != "record_public_physical_slot_guard_only_for_untimed_warmup"
                        or any(warmup[k] != v for k, v in status.items())
                        or status["reason"] == "unexplained_early_stop"
                        or any(
                            b["tokens"] != b["verifier_argmax"]
                            or len(b["tokens"]) != b["accepted_draft_tokens"] + 1
                            or not 0 <= b["accepted_draft_tokens"] <= 8
                            for b in warmup["trace"]
                        )
                    ):
                        raise ValueError(
                            "warmup termination or actual verifier proof differs"
                        )
                tokens, raw = committed_tokens(
                    row["verifier_trace"], config["max_new_tokens"], 151645
                )
                if (
                    tokens != row["output_ids"]
                    or raw != row["raw_output_ids"]
                    or any(
                        b["tokens"] != b["verifier_argmax"]
                        for b in row["verifier_trace"]
                    )
                ):
                    raise ValueError(
                        "SD-square output differs from observed verification"
                    )
            if (
                row["output_tokens"] != len(row["output_ids"])
                or not 0 < row["output_tokens"] <= config["max_new_tokens"]
            ):
                raise ValueError("SD-square counted output differs")
            if (
                row["output_hash"]
                != hashlib.sha256(
                    struct.pack(f"<{len(row['output_ids'])}i", *row["output_ids"])
                ).hexdigest()
            ):
                raise ValueError("SD-square token hash differs from recorded IDs")
            accepted = [b["accepted_draft_tokens"] for b in row["verifier_trace"]]
            if (
                row["accepted_draft_lengths"] != accepted
                or row["acceptance_lengths"] != [n + 1 for n in accepted]
                or row["proposal_lengths"] != [8] * len(accepted)
            ):
                raise ValueError(
                    "SD-square acceptance summary differs from verifier trace"
                )
            if (
                row["output_tokens"] < config["max_new_tokens"]
                and row["output_ids"][-1] != 151645
            ):
                raise ValueError("SD-square output ended without EOS or cap")
            if row["steering_checkpoint_sha256"] != value.get(
                "checkpoint_sha256"
            ) or row["steering_trainable_sha256"] != value.get("trainable_sha256"):
                raise ValueError("SD-square decoded another steering checkpoint")
            if row["steering_inference_sha256"] != identities.get(
                row["method"], {}
            ).get("inference_sha256"):
                raise ValueError(
                    "SD-square row differs from verified inference weights"
                )
        for p in (worker_path, path):
            inputs[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        records.extend(rows)
    if len(groups) != config["requests"] or any(
        set(g) != set(config["variants"]) for g in groups.values()
    ):
        raise ValueError("SD-square campaign lacks paired methods")
    if any(
        len({tuple(r["input_ids"]) for r in g.values()}) != 1 for g in groups.values()
    ):
        raise ValueError("SD-square methods received different prompts")
    (output / "completion-gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "config_sha256": config_sha,
                "records": len(records),
                "requests": len(groups),
                "methods": list(config["variants"]),
                "input_sha256": inputs,
                "inference_precision": config["inference_precision"],
                "steering_fingerprints": fingerprints,
                "ar_exact_counts": {
                    m: sum(
                        g[m]["output_ids"] == g["native_ar"]["output_ids"]
                        for g in groups.values()
                    )
                    for m in config["variants"]
                },
                "scope": config["scope"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
