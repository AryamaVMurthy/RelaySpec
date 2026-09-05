from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from relayspec.gates import select_profile_provider

EXPECTED_REQUESTS = {
    "gsm8k": 128,
    "humaneval": 164,
    "mbpp": 378,
    "mtbench": 160,
}
EXPECTED_CLUSTERS = {**EXPECTED_REQUESTS, "mtbench": 80}
SOURCE_REGIONS = {"prefill_source_trunk", "verification_source_trunk"}
CYCLE_REGIONS = {"draft", "verification_full_target"}
RELAY_REGIONS = {"prefill_relay", "relay"}


def amdahl_diagnostics(
    source_metrics: dict[str, Any], candidate_metrics: dict[str, Any]
) -> dict[str, float] | None:
    source_regions = source_metrics.get("profile_region_totals_ms")
    candidate_regions = candidate_metrics.get("profile_region_totals_ms")
    if not isinstance(source_regions, dict) or not isinstance(candidate_regions, dict):
        return None
    reference_seconds = float(source_metrics["output_tokens"]) / float(
        source_metrics["end_to_end_tokens_per_second"]
    )
    source_seconds = sum(
        float(source_regions.get(name, 0.0)) for name in SOURCE_REGIONS
    )
    cycle_seconds = sum(float(source_regions.get(name, 0.0)) for name in CYCLE_REGIONS)
    relay_seconds = sum(
        float(candidate_regions.get(name, 0.0)) for name in RELAY_REGIONS
    )
    source_fraction = source_seconds / 1000.0 / reference_seconds
    cycle_fraction = cycle_seconds / 1000.0 / reference_seconds
    relay_fraction = relay_seconds / 1000.0 / reference_seconds
    other_fraction = 1.0 - source_fraction - cycle_fraction
    if min(source_fraction, cycle_fraction, relay_fraction, other_fraction) < -1e-6:
        raise ValueError("profile regions do not form non-negative Amdahl components")
    other_fraction = max(0.0, other_fraction)

    def cycle_micro_acceptance(metrics: dict[str, Any]) -> float:
        survival = metrics.get("acceptance_survival_by_position")
        if isinstance(survival, dict) and survival:
            return sum(float(value) for value in survival.values())
        return float(metrics["mean_acceptance_length"])

    acceptance_retention = cycle_micro_acceptance(
        candidate_metrics
    ) / cycle_micro_acceptance(source_metrics)
    modeled_latency = (
        cycle_fraction / acceptance_retention + other_fraction + relay_fraction
    )
    break_even_denominator = 1.0 - other_fraction - relay_fraction
    if modeled_latency <= 0.0 or break_even_denominator <= 0.0:
        raise ValueError("invalid Amdahl denominator")
    return {
        "source_fraction": source_fraction,
        "cycle_fraction": cycle_fraction,
        "other_fraction": other_fraction,
        "relay_fraction": relay_fraction,
        "acceptance_retention": acceptance_retention,
        "predicted_speedup": 1.0 / modeled_latency,
        "ideal_equal_acceptance_speedup": 1.0
        / (1.0 - source_fraction + relay_fraction),
        "break_even_acceptance_retention": cycle_fraction / break_even_denominator,
    }


def build_cell_summary(
    *,
    family: str,
    target: str,
    paper: dict[str, Any],
    evalplus: dict[str, Any] | None,
    benchmark_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reference = str(paper["reference_method"])
    result: dict[str, Any] = {
        "family": family,
        "target": target,
        "reference_method": reference,
        "benchmarks": {},
    }
    benchmark_summaries = dict(paper["by_benchmark"])
    benchmark_summaries.update(benchmark_overrides or {})
    for benchmark, summary in sorted(benchmark_summaries.items()):
        if benchmark not in EXPECTED_REQUESTS:
            continue
        requests = int(summary["paired_requests"])
        clusters = int(summary.get("bootstrap_clusters", requests))
        if requests != EXPECTED_REQUESTS[benchmark]:
            raise ValueError(
                f"{benchmark}: expected {EXPECTED_REQUESTS[benchmark]} requests, "
                f"found {requests}"
            )
        if clusters != EXPECTED_CLUSTERS[benchmark]:
            raise ValueError(
                f"{benchmark}: expected {EXPECTED_CLUSTERS[benchmark]} "
                f"bootstrap clusters, found {clusters}"
            )
        methods = summary["methods"]
        candidates = [name for name in methods if name != reference]
        if len(candidates) != 1:
            raise ValueError(
                f"{benchmark}: expected one candidate beside {reference}, "
                f"found {candidates}"
            )
        candidate = candidates[0]
        source_metrics = methods[reference]
        candidate_metrics = methods[candidate]
        row: dict[str, Any] = {
            "requests": requests,
            "bootstrap_clusters": clusters,
            "candidate_method": candidate,
            "source_tokens_per_second": source_metrics["end_to_end_tokens_per_second"],
            "relay_tokens_per_second": candidate_metrics[
                "end_to_end_tokens_per_second"
            ],
            "speedup": candidate_metrics["end_to_end_speedup_vs_reference"],
            "speedup_ci95": candidate_metrics["end_to_end_speedup_vs_reference_ci95"],
            "exact_match_rate": candidate_metrics[
                "exact_sequence_match_rate_vs_reference"
            ],
            "acceptance_retention": (
                candidate_metrics["mean_acceptance_length"]
                / source_metrics["mean_acceptance_length"]
            ),
            "source_cap_hit_rate": source_metrics.get("cap_hit_rate"),
            "relay_cap_hit_rate": candidate_metrics.get("cap_hit_rate"),
            "source_latency_p50_seconds": source_metrics.get(
                "request_latency_p50_seconds"
            ),
            "relay_latency_p50_seconds": candidate_metrics.get(
                "request_latency_p50_seconds"
            ),
            "source_latency_p95_seconds": source_metrics.get(
                "request_latency_p95_seconds"
            ),
            "relay_latency_p95_seconds": candidate_metrics.get(
                "request_latency_p95_seconds"
            ),
        }
        amdahl = amdahl_diagnostics(source_metrics, candidate_metrics)
        if amdahl is not None:
            row["amdahl"] = amdahl
            speed_ci_lower = float(row["speedup_ci95"][0])
            selected_kind = select_profile_provider(
                predicted_speedup=float(amdahl["predicted_speedup"]),
                speed_ci_lower=speed_ci_lower,
            )
            selects_relay = selected_kind == "relay"
            point_direction_match = (float(amdahl["predicted_speedup"]) > 1.0) == (
                float(row["speedup"]) > 1.0
            )
            row["profile_policy"] = {
                "selected_provider": candidate if selects_relay else reference,
                "realized_speedup": float(row["speedup"]) if selects_relay else 1.0,
                "direction_match": point_direction_match,
                "point_direction_match": point_direction_match,
                "speed_evidence_passes": speed_ci_lower > 1.0,
                "speed_ci_lower": speed_ci_lower,
                "predicted_margin_over_source": float(amdahl["predicted_speedup"])
                - 1.0,
            }
        if benchmark == "gsm8k" and "accuracy" in source_metrics:
            row["official_quality"] = {
                reference: {"accuracy": source_metrics["accuracy"]},
                candidate: {"accuracy": candidate_metrics["accuracy"]},
            }
        elif benchmark in {"humaneval", "mbpp"} and evalplus is not None:
            row["official_quality"] = evalplus["benchmarks"][benchmark]
            paired_quality = evalplus.get("paired_quality", {}).get(benchmark, {})
            if candidate in paired_quality:
                row["paired_quality"] = paired_quality[candidate]
        result["benchmarks"][benchmark] = row
    benchmark_rows = list(result["benchmarks"].values())
    if benchmark_rows:
        amdahl_rows = [row for row in benchmark_rows if "amdahl" in row]
        result["summary"] = {
            "geometric_mean_speedup": math.exp(
                sum(math.log(float(row["speedup"])) for row in benchmark_rows)
                / len(benchmark_rows)
            ),
            "positive_speed_ci_cells": sum(
                float(row["speedup_ci95"][0]) > 1.0 for row in benchmark_rows
            ),
            "benchmark_cells": len(benchmark_rows),
            "amdahl_direction_matches": sum(
                (float(row["amdahl"]["predicted_speedup"]) > 1.0)
                == (float(row["speedup"]) > 1.0)
                for row in amdahl_rows
            ),
            "amdahl_cells": len(amdahl_rows),
            "amdahl_mean_absolute_relative_error": (
                sum(
                    abs(
                        float(row["amdahl"]["predicted_speedup"])
                        / float(row["speedup"])
                        - 1.0
                    )
                    for row in amdahl_rows
                )
                / len(amdahl_rows)
                if amdahl_rows
                else None
            ),
            "profile_policy_geometric_mean_speedup": (
                math.exp(
                    sum(
                        math.log(float(row["profile_policy"]["realized_speedup"]))
                        for row in amdahl_rows
                    )
                    / len(amdahl_rows)
                )
                if amdahl_rows
                else None
            ),
            "profile_policy_direction_matches": sum(
                bool(row["profile_policy"]["direction_match"]) for row in amdahl_rows
            ),
        }
    return result


def aggregate_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the complete family/target/task matrix without reweighting.

    Every benchmark cell has one vote.  This summary is descriptive rather
    than a claim that the four task distributions occur equally in service.
    """
    rows = [row for cell in cells for row in cell["benchmarks"].values()]
    if not rows:
        raise ValueError("cannot aggregate an empty breadth matrix")
    amdahl_rows = [row for row in rows if "amdahl" in row and "profile_policy" in row]
    return {
        "benchmark_cells": len(rows),
        "raw_geometric_mean_speedup": math.exp(
            sum(math.log(float(row["speedup"])) for row in rows) / len(rows)
        ),
        "positive_speed_ci_cells": sum(
            float(row["speedup_ci95"][0]) > 1.0 for row in rows
        ),
        "amdahl_cells": len(amdahl_rows),
        "amdahl_direction_matches": sum(
            bool(row["profile_policy"]["direction_match"]) for row in amdahl_rows
        ),
        "amdahl_mean_absolute_relative_error": (
            sum(
                abs(
                    float(row["amdahl"]["predicted_speedup"]) / float(row["speedup"])
                    - 1.0
                )
                for row in amdahl_rows
            )
            / len(amdahl_rows)
            if amdahl_rows
            else None
        ),
        "profile_policy_geometric_mean_speedup": (
            math.exp(
                sum(
                    math.log(float(row["profile_policy"]["realized_speedup"]))
                    for row in amdahl_rows
                )
                / len(amdahl_rows)
            )
            if amdahl_rows
            else None
        ),
    }


def render_markdown(
    cells: list[dict[str, Any]], aggregate: dict[str, Any] | None = None
) -> str:
    lines = [
        "# RelaySpec frozen cross-task breadth matrix",
        "",
        "| Family | Target | Task | N/clusters | Source tok/s | Relay tok/s | Speedup [95% CI] | Exact | Acceptance retained |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        for benchmark, row in cell["benchmarks"].items():
            low, high = row["speedup_ci95"]
            lines.append(
                f"| {cell['family']} | {cell['target']} | {benchmark} | "
                f"{row['requests']}/{row['bootstrap_clusters']} | "
                f"{row['source_tokens_per_second']:.3f} | "
                f"{row['relay_tokens_per_second']:.3f} | "
                f"{row['speedup']:.4f}x [{low:.4f}, {high:.4f}] | "
                f"{row['exact_match_rate']:.2%} | "
                f"{row['acceptance_retention']:.2%} |"
            )
    lines.extend(
        [
            "",
            "Both MT-Bench turns are one bootstrap cluster. Official code quality is stored in the JSON artifact; MT-Bench has no quality claim without a judge run.",
            "",
            "## Cell summaries",
            "",
            "| Family | Target | Geometric-mean speedup | Positive 95% CI cells | Amdahl direction matches | Amdahl mean relative error |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for cell in cells:
        summary = cell["summary"]
        amdahl_error = summary["amdahl_mean_absolute_relative_error"]
        amdahl_error_text = "n/a" if amdahl_error is None else f"{amdahl_error:.2%}"
        lines.append(
            f"| {cell['family']} | {cell['target']} | "
            f"{summary['geometric_mean_speedup']:.4f}x | "
            f"{summary['positive_speed_ci_cells']}/{summary['benchmark_cells']} | "
            f"{summary['amdahl_direction_matches']}/{summary['amdahl_cells']} | "
            f"{amdahl_error_text} |"
        )
    if aggregate is not None:
        lines.extend(
            [
                "",
                "## Complete matrix summary",
                "",
                "The geometric means give every family/target/task cell equal weight; they are descriptive and do not assume an operational workload mixture.",
                "",
                f"- Raw RelaySpec geometric-mean speedup: {aggregate['raw_geometric_mean_speedup']:.4f}x.",
                f"- Cells with paired 95% speed interval above one: {aggregate['positive_speed_ci_cells']}/{aggregate['benchmark_cells']}.",
                f"- Amdahl direction matches: {aggregate['amdahl_direction_matches']}/{aggregate['amdahl_cells']}.",
                f"- Amdahl mean absolute relative error: {aggregate['amdahl_mean_absolute_relative_error']:.2%}.",
                f"- Descriptive profile-policy geometric-mean speedup: {aggregate['profile_policy_geometric_mean_speedup']:.4f}x.",
            ]
        )
    lines.extend(
        [
            "",
            "## Automatic coefficient-free profile policy",
            "",
            "The policy selects RelaySpec only when the measured Amdahl prediction and the paired 95% speed lower bound both exceed the source boundary of one; otherwise it retains source reuse. No quality proxy or tuned threshold enters the decision.",
            "",
            "| Family | Target | Task | Predicted | Selected provider | Realized speedup | Direction match |",
            "|---|---|---|---:|---|---:|---:|",
        ]
    )
    for cell in cells:
        for benchmark, row in cell["benchmarks"].items():
            policy = row.get("profile_policy")
            if policy is None:
                continue
            lines.append(
                f"| {cell['family']} | {cell['target']} | {benchmark} | "
                f"{row['amdahl']['predicted_speedup']:.4f}x | "
                f"{policy['selected_provider']} | "
                f"{policy['realized_speedup']:.4f}x | "
                f"{'yes' if policy['direction_match'] else 'no'} |"
            )
    lines.extend(
        [
            "",
            "## Measured Amdahl mechanism check",
            "",
            "| Family | Target | Task | Source share | Relay/reference | Retention | Break-even | Predicted | Observed | Equal-acceptance ceiling |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in cells:
        for benchmark, row in cell["benchmarks"].items():
            amdahl = row.get("amdahl")
            if amdahl is None:
                continue
            lines.append(
                f"| {cell['family']} | {cell['target']} | {benchmark} | "
                f"{amdahl['source_fraction']:.2%} | "
                f"{amdahl['relay_fraction']:.2%} | "
                f"{amdahl['acceptance_retention']:.2%} | "
                f"{amdahl['break_even_acceptance_retention']:.2%} | "
                f"{amdahl['predicted_speedup']:.4f}x | "
                f"{row['speedup']:.4f}x | "
                f"{amdahl['ideal_equal_acceptance_speedup']:.4f}x |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell",
        nargs=3,
        action="append",
        metavar=("FAMILY", "TARGET", "RESULT_DIR"),
        required=True,
    )
    parser.add_argument(
        "--benchmark-override",
        nargs=4,
        action="append",
        metavar=("FAMILY", "TARGET", "BENCHMARK", "RESULT_DIR"),
        help="Replace one benchmark summary with a complete dedicated run.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    overrides: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for family, target, benchmark, raw_path in args.benchmark_override or []:
        key = (family, target)
        if benchmark in overrides.setdefault(key, {}):
            raise ValueError(f"duplicate override for {family}/{target}/{benchmark}")
        override_paper = json.loads(
            (Path(raw_path) / "benchmark-paper-summary.json").read_text(
                encoding="utf-8"
            )
        )
        if benchmark not in override_paper["by_benchmark"]:
            raise ValueError(f"override directory does not contain {benchmark}")
        overrides[key][benchmark] = override_paper["by_benchmark"][benchmark]

    cells = []
    for family, target, raw_path in args.cell:
        result_dir = Path(raw_path)
        paper = json.loads(
            (result_dir / "benchmark-paper-summary.json").read_text(encoding="utf-8")
        )
        evalplus_path = result_dir / "evalplus-summary.json"
        evalplus = (
            json.loads(evalplus_path.read_text(encoding="utf-8"))
            if evalplus_path.exists()
            else None
        )
        cells.append(
            build_cell_summary(
                family=family,
                target=target,
                paper=paper,
                evalplus=evalplus,
                benchmark_overrides=overrides.get((family, target)),
            )
        )
    aggregate = aggregate_summary(cells)
    payload = {"cells": cells, "aggregate": aggregate}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(
        render_markdown(cells, aggregate=aggregate), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
