"""Reconstruct the SD-square first-divergence diagnosis from saved verifier traces."""

import argparse
import hashlib
import json
from pathlib import Path

from relayspec.sd_square_adapter import committed_tokens


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/submission/baselines/sd-square-numerical-replay.json"),
    )
    args = parser.parse_args()
    config_path = args.config
    config = json.loads(config_path.read_text())
    study = "sd-square-causal" if "causal_probe" in config else "sd-square-numerical"
    ledger_path = Path("reports/mapper-scaling-20260905") / study / "jobs.json"
    source_path = args.run / "source-commit.txt"
    if (
        source_path.read_text().strip()
        != json.loads(ledger_path.read_text())["source_commit"]
    ):
        raise ValueError("SD-square replay source provenance differs")
    prior_row_path = args.prior / "decoding-rank2.json"
    prior_gate_path = args.prior / "pilot-rank2.json"
    if (
        digest(prior_row_path) != config["source_decoding_sha256"]
        or digest(prior_gate_path) != config["source_gate_sha256"]
    ):
        raise ValueError("SD-square original failed run changed")
    prior_row, prior_gate = (
        json.loads(prior_row_path.read_text()),
        json.loads(prior_gate_path.read_text()),
    )
    if (
        prior_gate["status"] != "fail"
        or prior_gate["ar_exact"]
        or prior_row["ar_exact"]
    ):
        raise ValueError("original failed AR identity was relabeled")
    gate_path = args.run / "sd-square-numerical-gate.json"
    gate = json.loads(gate_path.read_text())
    if (
        gate["status"] != "pass"
        or gate["original_ar_identity_gate"] != "failed_preserved"
    ):
        raise ValueError("numerical diagnosis completion is missing")
    inputs = {
        str(p): digest(p)
        for p in (
            config_path,
            prior_row_path,
            prior_gate_path,
            gate_path,
            ledger_path,
            source_path,
        )
    }
    rows = []
    for rank in range(4):
        path = args.run / f"numerical-rank{rank}.json"
        row = json.loads(path.read_text())
        if (
            digest(path) != gate["input_sha256"][path.name]
            or row["rank"] != rank
            or row["config_sha256"] != digest(config_path)
            or row["source_gate_sha256"] != digest(prior_gate_path)
            or row["source_decoding_sha256"] != digest(prior_row_path)
            or row["target_attention"] != config["target_attention_by_rank"][rank]
        ):
            raise ValueError("SD-square numerical provenance differs")
        tokens, raw = committed_tokens(row["sd_trace"], 64, 151645)
        if (
            tokens != row["sd_ids"]
            or raw != row["raw_committed_ids"]
            or any(b["tokens"] != b["verifier_argmax"] for b in row["sd_trace"])
        ):
            raise ValueError("SD-square committed tokens differ from their verifier")
        if rank < 2 and (
            tokens != prior_row["output_ids"] or row["ar_ids"] != prior_row["ar_ids"]
        ):
            raise ValueError("original SDPA difference did not reproduce")
        difference = next(
            (i for i, (a, b) in enumerate(zip(row["ar_ids"], tokens)) if a != b), None
        )
        detail = row["first_difference"]
        if (
            difference != config["first_observed_difference"]
            or detail["index"] != difference
        ):
            raise ValueError("first SD-square difference moved")
        block = next(
            b
            for b in row["sd_trace"]
            if b["output_start"] <= difference < b["output_start"] + len(b["tokens"])
        )
        offset = difference - block["output_start"]
        prefix = block["valid_cached_prefix_ids"] + block["query_ids"][: offset + 1]
        if (
            prefix != prior_row["input_ids"] + tokens[:difference]
            or block["query_positions"][offset] != len(prefix) - 1
            or detail["prefix_length"] != len(prefix)
            or detail["ar_token"] != row["ar_ids"][difference]
            or detail["sd_token"] != tokens[difference]
            or detail["ar_top_ids"] != row["ar_trace"][difference]["top_ids"]
            or detail["ar_top_scores"] != row["ar_trace"][difference]["top_scores"]
            or detail["sd_top_ids"] != block["top_ids"][offset]
            or detail["sd_top_scores"] != block["top_scores"][offset]
            or gate["first_differences"][row["target_attention"]] != detail
        ):
            raise ValueError("SD-square same-prefix logit diagnosis does not reproduce")
        public_matches = row["public_ar_first16"] == row["ar_ids"][:16]
        if (
            public_matches != row["public_ar_matches_correct_cached_ar"]
            or public_matches
            != gate["public_ar_matches_correct_cached_ar"][row["target_attention"]]
        ):
            raise ValueError("public AR helper comparison differs")
        if "causal_probe" in config:
            causal = row["causal_probe"]
            actual = next(
                b
                for b in row["sd_trace"]
                if b["physical_cache_prefix"] == causal["physical_cache_prefix"]
            )
            offset = causal["query_offset"]
            if (
                causal["status"] != "pass"
                or causal["physical_cache_prefix"]
                != config["causal_probe"]["physical_cache_prefix"]
                or offset != config["causal_probe"]["query_offset"]
                or causal["original_query_ids"][: offset + 1]
                != causal["changed_query_ids"][: offset + 1]
                or causal["original_query_ids"][offset + 1 :]
                == causal["changed_query_ids"][offset + 1 :]
                or causal["original_query_ids"] != actual["query_ids"]
                or causal["baseline_logits_float32_sha256"]
                != actual["logits_float32_sha256"]
                or not all(
                    causal[k]
                    for k in (
                        "cache_copies_exact_and_independent",
                        "original_cache_unchanged",
                        "prefix_logits_bit_identical",
                    )
                )
                or gate["causal_probes"][row["target_attention"]] != causal
            ):
                raise ValueError("causal intervention evidence changed")
        inputs[str(path)] = digest(path)
        rows.append(row)
    for a, b in ((0, 1), (2, 3)):
        if {k: v for k, v in rows[a].items() if k != "rank"} != {
            k: v for k, v in rows[b].items() if k != "rank"
        }:
            raise ValueError("duplicate numerical observations differ")
    args.output.write_text(
        json.dumps(
            {
                "status": "complete_numerical_diagnostic",
                "input_sha256": inputs,
                "first_differences": gate["first_differences"],
                "public_ar_matches_correct_cached_ar": gate[
                    "public_ar_matches_correct_cached_ar"
                ],
                "original_ar_identity_gate": "failed_preserved",
                **(
                    {"causal_probes": gate["causal_probes"]}
                    if "causal_probe" in config
                    else {}
                ),
                "scope": "Two exact replicas per attention backend. Same effective token prefix and "
                "logical position at the first differing choice. Public AR helper mismatch is observed "
                "separately. This does not isolate cache history, query shape, or an exact CUDA kernel. "
                + (
                    "Changing future query tokens preserves earlier verifier logits with exact copied caches. "
                    if "causal_probe" in config
                    else "Causal-mask intervention remains outstanding. "
                )
                + "Task-quality comparison remains outstanding. No new fitting.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
