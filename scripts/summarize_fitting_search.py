"""Summarize completed, matched data and optimizer-budget evaluations."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from relayspec.ar_paper_evidence import read_rows


def load_cell(path):
    gate = json.loads((path / "completion-gate.json").read_text())
    analysis = json.loads((path / "analysis.json").read_text())
    assert gate["status"] == "pass" and analysis["status"] == "complete"
    for name, expected in analysis["raw_sha256"].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"raw evidence changed: {path / name}")
    rows = read_rows(path)
    if len(rows) != gate["records"]:
        raise ValueError("incomplete evaluation")
    metrics = analysis["tasks"]["math500"]
    relay = metrics["methods"]["relay_p"]
    native = metrics["native_reference"]["methods"]["relay_p"]
    checkpoint = json.loads((path / "controlled-metrics.json").read_text())[
        "checkpoint"
    ]
    pairs = {}
    for row in rows:
        key = (row["problem_id"], row.get("repetition", 0))
        pair = pairs.setdefault(key, {})
        pair[row["method"]] = row
    keys = sorted(pairs)
    # AR output identity checks the exact evaluated request set and generation.
    fingerprint = hashlib.sha256(
        json.dumps(
            [(key, pairs[key]["native_ar"]["output_hash"]) for key in keys]
        ).encode()
    ).hexdigest()
    return {
        "path": str(path),
        "distinct": checkpoint["data_accounting"]["distinct_records_seen"],
        "presentations": checkpoint["data_accounting"]["record_presentations"],
        "updates": checkpoint["step"],
        "training_seconds": checkpoint["training_seconds"],
        "tokens_per_second": relay["tokens_per_second"],
        "speedup_vs_ar": relay["throughput_ratio"],
        "speedup_ci95": relay["throughput_ci95"],
        "native_retained": native["throughput_ratio"],
        "acceptance": relay["proposed_token_acceptance"],
        "progress": relay["progress_per_cycle"],
        "correct": round(relay["accuracy"] * metrics["requests"]),
        "ar_correct": round(
            metrics["methods"]["native_ar"]["accuracy"] * metrics["requests"]
        ),
        "requests": metrics["requests"],
        "request_fingerprint": fingerprint,
    }, np.array(
        [
            [
                pairs[key]["relay_p"]["output_tokens"],
                pairs[key]["relay_p"]["request_seconds"],
            ]
            for key in keys
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    old = Path("reports/controlled-scaling-20260905")
    new = Path("reports/unfitted-controls-20260905")
    fixed = []
    arrays = []
    for n in [64, 128, 256, 512, 1024, 2048, 4096]:
        root = new if n < 512 else old
        run = "rs-continuous" if n == 4096 else f"rs-data{n}"
        path = root / run / "evaluation-step-001024"
        if not (path / "analysis.json").exists():
            if args.require_complete:
                raise ValueError(f"missing fixed-data cell: {path}")
            continue
        cell, array = load_cell(path)
        if (
            cell["updates"] != 1024
            or cell["presentations"] != 4096
            or cell["distinct"] != n
        ):
            raise ValueError("data budget mismatch")
        fixed.append(cell)
        arrays.append(array)
    if len({cell["request_fingerprint"] for cell in fixed}) != 1:
        raise ValueError("fixed-data evaluations do not share identical AR outputs")
    best = max(range(len(fixed)), key=lambda i: fixed[i]["tokens_per_second"])
    rng = np.random.default_rng(1729)
    indices = rng.integers(0, len(arrays[best]), size=(10000, len(arrays[best])))
    best_sums = arrays[best][indices].sum(axis=1)
    for cell, array in zip(fixed, arrays, strict=True):
        sums = array[indices].sum(axis=1)
        ratios = (sums[:, 0] / sums[:, 1]) / (best_sums[:, 0] / best_sums[:, 1])
        cell["throughput_fraction_of_observed_best"] = (
            cell["tokens_per_second"] / fixed[best]["tokens_per_second"]
        )
        cell["fraction_ci95_conditional_on_selected_reference"] = np.quantile(
            ratios, [0.025, 0.975]
        ).tolist()
        cell["within_five_percent_point_estimate"] = (
            cell["throughput_fraction_of_observed_best"] >= 0.95
        )
    continuous = []
    for step in [128, 256, 512, 1024, 2048]:
        path = old / "rs-continuous" / f"evaluation-step-{step:06d}"
        if (path / "analysis.json").exists():
            continuous.append(load_cell(path)[0])
        elif args.require_complete:
            raise ValueError(f"missing continuous cell: {path}")
    report = {
        "complete": len(fixed) == 7 and len(continuous) == 5,
        "scope": "One fitting seed on 128 development-exposed requests. Best and threshold are descriptive selections. Cross-run paired intervals condition on the observed-best reference, omit fitting-seed variation, and do not adjust for selecting that reference. They are not a confirmatory noninferiority test.",
        "fixed_data": fixed,
        "continuous": continuous,
        "observed_best_distinct": fixed[best]["distinct"],
        "smallest_tested_within_five_percent": min(
            c["distinct"] for c in fixed if c["within_five_percent_point_estimate"]
        ),
    }
    (new / "fitting-search-summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"fixed_data", "continuous", "scope"}
            }
        )
    )


if __name__ == "__main__":
    main()
