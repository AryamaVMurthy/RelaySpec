from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from relayspec.benchmarking import paired_bootstrap_speedup
from relayspec.cost_model import AmdahlInputs, estimate_relay_speedup

SOURCE_REGIONS = {"prefill_source_trunk", "verification_source_trunk"}
CYCLE_REGIONS = {"draft", "verification_full_target"}
RELAY_REGIONS = {"prefill_relay", "relay"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-method", required=True)
    parser.add_argument("--candidate-method", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--subset-manifest",
        type=Path,
        help="Optional preregistered manifest used to filter problem IDs.",
    )
    return parser.parse_args()


def sum_regions(rows: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in rows:
        for name, milliseconds in row.get("profile_regions_ms", {}).items():
            result[name] = result.get(name, 0.0) + float(milliseconds) / 1_000.0
    return result


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for path in args.input
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if args.subset_manifest is not None:
        subset = json.loads(args.subset_manifest.read_text(encoding="utf-8"))
        expected_ids = {str(row["problem_id"]) for row in subset["records"]}
        rows = [row for row in rows if str(row["problem_id"]) in expected_ids]
        observed_ids = {str(row["problem_id"]) for row in rows}
        if observed_ids != expected_ids:
            raise RuntimeError("profile rows do not cover the subset manifest")
    reference = [row for row in rows if row["method"] == args.reference_method]
    candidate = [row for row in rows if row["method"] == args.candidate_method]
    if not reference or len(reference) != len(candidate):
        raise RuntimeError("profile requires equally sized non-empty method samples")

    reference_by_key = {
        (row["problem_id"], int(row["repetition"])): row for row in reference
    }
    candidate_by_key = {
        (row["problem_id"], int(row["repetition"])): row for row in candidate
    }
    if reference_by_key.keys() != candidate_by_key.keys():
        raise RuntimeError("profile methods are not request paired")

    reference_seconds = sum(float(row["request_seconds"]) for row in reference)
    candidate_seconds = sum(float(row["request_seconds"]) for row in candidate)
    reference_regions = sum_regions(reference)
    candidate_regions = sum_regions(candidate)
    source_seconds = sum(reference_regions.get(name, 0.0) for name in SOURCE_REGIONS)
    cycle_seconds = sum(reference_regions.get(name, 0.0) for name in CYCLE_REGIONS)
    other_seconds = reference_seconds - source_seconds - cycle_seconds
    relay_seconds = sum(candidate_regions.get(name, 0.0) for name in RELAY_REGIONS)
    if min(source_seconds, cycle_seconds, other_seconds, relay_seconds) < 0:
        raise RuntimeError("component regions do not form a valid non-negative profile")

    reference_acceptances = [
        int(value) for row in reference for value in row["acceptance_lengths"]
    ]
    candidate_acceptances = [
        int(value) for row in candidate for value in row["acceptance_lengths"]
    ]
    reference_acceptance = sum(reference_acceptances) / len(reference_acceptances)
    candidate_acceptance = sum(candidate_acceptances) / len(candidate_acceptances)
    inputs = AmdahlInputs(
        cycle_fraction=cycle_seconds / reference_seconds,
        source_fraction=source_seconds / reference_seconds,
        other_fraction=other_seconds / reference_seconds,
        relay_fraction=relay_seconds / reference_seconds,
        source_committed_per_cycle=reference_acceptance,
        relay_committed_per_cycle=candidate_acceptance,
    )
    estimate = estimate_relay_speedup(inputs)
    interval = paired_bootstrap_speedup(
        rows,
        reference_method=args.reference_method,
        candidate_method=args.candidate_method,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    exact_matches = sum(
        reference_by_key[key]["output_hash"] == candidate_by_key[key]["output_hash"]
        for key in reference_by_key
    )
    result = {
        "reference_method": args.reference_method,
        "candidate_method": args.candidate_method,
        "paired_requests": len(reference),
        "exact_sequence_matches": exact_matches,
        "exact_sequence_match_rate": exact_matches / len(reference),
        "reference_request_seconds": reference_seconds,
        "candidate_request_seconds": candidate_seconds,
        "observed_speedup": reference_seconds / candidate_seconds,
        "paired_bootstrap_95_percent": interval,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "reference_acceptance_length_micro": reference_acceptance,
        "candidate_acceptance_length_micro": candidate_acceptance,
        "acceptance_retention": candidate_acceptance / reference_acceptance,
        "reference_regions_seconds": reference_regions,
        "candidate_regions_seconds": candidate_regions,
        "amdahl_inputs": inputs.__dict__,
        "amdahl_estimate": estimate.__dict__,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = f"""# Optimized source-reuse profile

This artifact is generated from immutable per-request rows. The reference is
`{args.reference_method}` and the candidate is `{args.candidate_method}`.

| Quantity | Result |
|---|---:|
| Paired requests | {len(reference)} |
| Exact sequence agreement | {exact_matches}/{len(reference)} ({exact_matches / len(reference):.1%}) |
| Reference request time | {reference_seconds:.3f} s |
| Relay request time | {candidate_seconds:.3f} s |
| Observed end-to-end speedup | **{reference_seconds / candidate_seconds:.3f}x** |
| Paired request-bootstrap 95% interval | [{interval["lower"]:.3f}, {interval["upper"]:.3f}]x |
| Source-trunk runtime share | {inputs.source_fraction:.1%} |
| Relay runtime relative to reference | {inputs.relay_fraction:.1%} |
| Acceptance retention | {estimate.acceptance_ratio:.1%} |
| Amdahl-modeled speedup | {estimate.predicted_speedup:.3f}x |
| Equal-acceptance ceiling | {estimate.ideal_equal_acceptance_speedup:.3f}x |
| Break-even acceptance retention | {estimate.break_even_acceptance_ratio:.1%} |

The bootstrap resamples whole paired requests, so prompt length and output-length
variation remain coupled across methods. The same-run Amdahl value reconstructs
latency from measured source, target/draft cycle, other, relay, and
micro-averaged committed-token components; it is a mechanism diagnostic, not
an independent prediction. A genuine held-out forecast must freeze these
quantities on development data before measuring the test workload.
"""
    args.output_md.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
