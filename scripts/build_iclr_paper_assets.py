#!/usr/bin/env python3
"""Build RelaySpec paper tables and vector figures from final JSON artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FAMILY_ORDER = {"dflash": 0, "eagle3": 1}
TARGET_ORDER = {"8b": 0, "14b": 1}
TASK_ORDER = {"gsm8k": 0, "humaneval": 1, "mbpp": 2, "mtbench": 3}
FAMILY_LABEL = {"dflash": "DFlash", "eagle3": "EAGLE-3"}
TARGET_LABEL = {"8b": "8B", "14b": "14B"}
TASK_LABEL = {
    "gsm8k": "GSM8K",
    "humaneval": "HumanEval",
    "mbpp": "MBPP",
    "mtbench": "MT-Bench",
}
SIMILARITY_TASK_ORDER = ("math500", "gsm8k", "humaneval", "mbpp", "mtbench")
FIXED_TIME = dt.datetime(2026, 8, 28, tzinfo=dt.UTC)
PDF_METADATA = {
    "Creator": "RelaySpec paper asset builder",
    "Producer": "Matplotlib",
    "CreationDate": FIXED_TIME,
    "ModDate": FIXED_TIME,
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _main_specifications() -> list[tuple[str, str, str, str, str]]:
    return [
        ("dflash", "8b", "dflash-8b-math500", "optimized_source_reuse", "relay_p"),
        ("dflash", "14b", "dflash-14b-math500", "optimized_source_reuse", "relay_p"),
        ("eagle3", "8b", "eagle3-8b-math500", "source_reuse_eagle3", "relay_eagle3"),
        ("eagle3", "14b", "eagle3-14b-math500", "source_reuse_eagle3", "relay_eagle3"),
    ]


def _profile_share(method: dict[str, Any], *keys: str) -> float:
    shares = method["profile_region_request_shares"]
    return sum(float(shares.get(key, 0.0)) for key in keys)


def _load_main_math(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, target, directory, source_name, relay_name in _main_specifications():
        path = root / "reports" / "final" / directory / "benchmark-paper-summary.json"
        raw = _read_json(path)
        group = raw["by_benchmark"]["math500"]
        source = group["methods"][source_name]
        relay = group["methods"][relay_name]
        requests = int(group["paired_requests"])
        source_trunk = _profile_share(
            source, "prefill_source_trunk", "verification_source_trunk"
        )
        target_work = _profile_share(
            source, "prefill_full_target", "verification_full_target"
        )
        proposer_work = _profile_share(source, "draft")
        other_work = max(0.0, 1.0 - source_trunk - target_work - proposer_work)
        relay_work = _profile_share(relay, "prefill_relay", "relay")
        relay_in_source_units = relay_work / float(
            relay["end_to_end_speedup_vs_reference"]
        )
        ideal_speedup = 1.0 / (1.0 - source_trunk + relay_in_source_units)
        rows.append(
            {
                "family_key": family,
                "family": FAMILY_LABEL[family],
                "target_key": target,
                "target": TARGET_LABEL[target],
                "requests": requests,
                "source_tps": float(source["end_to_end_tokens_per_second"]),
                "relay_tps": float(relay["end_to_end_tokens_per_second"]),
                "speedup": float(relay["end_to_end_speedup_vs_reference"]),
                "speedup_ci": [
                    float(value)
                    for value in relay["end_to_end_speedup_vs_reference_ci95"]
                ],
                "source_accuracy": float(source["accuracy"]),
                "relay_accuracy": float(relay["accuracy"]),
                "exact_rate": float(relay["exact_sequence_match_rate_vs_reference"]),
                "exact_matches": round(
                    requests * float(relay["exact_sequence_match_rate_vs_reference"])
                ),
                "source_acceptance": float(source["mean_acceptance_length"]),
                "relay_acceptance": float(relay["mean_acceptance_length"]),
                "acceptance_retention": float(relay["mean_acceptance_length"])
                / float(source["mean_acceptance_length"]),
                "source_trunk_fraction": source_trunk,
                "target_fraction": target_work,
                "proposer_fraction": proposer_work,
                "other_fraction": other_work,
                "relay_fraction_of_relay": relay_work,
                "relay_fraction_in_source_units": relay_in_source_units,
                "ideal_speedup": ideal_speedup,
                "source_survival": {
                    int(position): float(value)
                    for position, value in source[
                        "acceptance_survival_by_position"
                    ].items()
                },
                "relay_survival": {
                    int(position): float(value)
                    for position, value in relay[
                        "acceptance_survival_by_position"
                    ].items()
                },
                "source_path": str(path.relative_to(root)),
            }
        )
    return rows


def _load_breadth(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = _read_json(root / "reports" / "final" / "BREADTH_MATRIX.json")
    rows: list[dict[str, Any]] = []
    cells = sorted(
        raw["cells"],
        key=lambda item: (FAMILY_ORDER[item["family"]], TARGET_ORDER[item["target"]]),
    )
    for cell in cells:
        for task, item in sorted(
            cell["benchmarks"].items(), key=lambda pair: TASK_ORDER[pair[0]]
        ):
            rows.append(
                {
                    "family_key": cell["family"],
                    "family": FAMILY_LABEL[cell["family"]],
                    "target_key": cell["target"],
                    "target": TARGET_LABEL[cell["target"]],
                    "task_key": task,
                    "task": TASK_LABEL[task],
                    "requests": int(item["requests"]),
                    "clusters": int(item["bootstrap_clusters"]),
                    "source_tps": float(item["source_tokens_per_second"]),
                    "relay_tps": float(item["relay_tokens_per_second"]),
                    "speedup": float(item["speedup"]),
                    "speedup_ci": [float(value) for value in item["speedup_ci95"]],
                    "exact_rate": float(item["exact_match_rate"]),
                    "acceptance_retention": float(
                        item["amdahl"]["acceptance_retention"]
                    ),
                    "break_even_retention": float(
                        item["amdahl"]["break_even_acceptance_retention"]
                    ),
                    "acceptance_margin": float(item["amdahl"]["acceptance_retention"])
                    - float(item["amdahl"]["break_even_acceptance_retention"]),
                    "source_fraction": float(item["amdahl"]["source_fraction"]),
                    "relay_fraction": float(item["amdahl"]["relay_fraction"]),
                    "accounting_speedup": float(item["amdahl"]["predicted_speedup"]),
                    "policy_provider": str(item["profile_policy"]["selected_provider"]),
                    "policy_pass": bool(
                        item["profile_policy"]["speed_evidence_passes"]
                    ),
                    "official_quality": item.get("official_quality", {}),
                }
            )
    return rows, raw["aggregate"]


def _load_training(root: Path) -> dict[tuple[str, str], float]:
    paths = {
        ("dflash", "8b"): root
        / "reports/training/dflash-relative-8b/relay-training-summary.json",
        ("dflash", "14b"): root
        / "reports/training/dflash-relative-14b/relay-training-summary.json",
        ("eagle3", "8b"): root
        / "reports/design-selection/eagle3/relayspec-eagle3-scale-full-25557/relay-training-summary.json",
        ("eagle3", "14b"): root
        / "reports/training/eagle3-14b/relay-training-summary.json",
    }
    return {
        key: float(_read_json(path)["elapsed_seconds"]) for key, path in paths.items()
    }


def _load_native_controls(root: Path) -> dict[tuple[str, str], dict[str, float | None]]:
    controls: dict[tuple[str, str], dict[str, float | None]] = {
        ("dflash", "14b"): {
            "native_ar_tps": 26.712121366212628,
            "native_proposer_tps": None,
        }
    }
    paths = {
        ("dflash", "8b"): root
        / "reports/d8-proposal-math500/benchmark-paper-summary.json",
        ("eagle3", "8b"): root
        / "reports/final/eagle3-8b-math500-native/benchmark-paper-summary.json",
        ("eagle3", "14b"): root
        / "reports/final/eagle3-14b-math500-native/benchmark-paper-summary.json",
    }
    method_names = {
        ("dflash", "8b"): ("native_ar", "native_target_dflash"),
        ("eagle3", "8b"): ("native_ar", "native_target_eagle3"),
        ("eagle3", "14b"): ("native_ar", "native_target_eagle3"),
    }
    for key, path in paths.items():
        methods = _read_json(path)["methods"]
        ar_name, proposer_name = method_names[key]
        controls[key] = {
            "native_ar_tps": float(methods[ar_name]["end_to_end_tokens_per_second"]),
            "native_proposer_tps": float(
                methods[proposer_name]["end_to_end_tokens_per_second"]
            ),
        }
    return controls


def _load_heldout_forecasts(root: Path) -> list[dict[str, Any]]:
    development = {
        ("dflash", "8b"): (
            root
            / "reports/design-selection/dflash-relative-8b/development-analysis.json",
            ("amdahl_estimate", "predicted_speedup"),
        ),
        ("dflash", "14b"): (
            root
            / "reports/design-selection/dflash-relative-14b/development-analysis.json",
            ("observed_speedup",),
        ),
        ("eagle3", "8b"): (
            root / "reports/eagle3-8b-scale-validation/analysis.json",
            ("amdahl_estimate", "predicted_speedup"),
        ),
        ("eagle3", "14b"): (
            root / "reports/eagle3-14b-scale-validation/analysis.json",
            ("amdahl_estimate", "predicted_speedup"),
        ),
    }
    heldout = {
        ("dflash", "8b"): (
            root
            / "reports/final/dflash-8b-math500/math500-confirmatory-paper-summary.json",
            "relay_p",
        ),
        ("dflash", "14b"): (
            root
            / "reports/final/dflash-14b-math500/math500-confirmatory-paper-summary.json",
            "relay_p",
        ),
        ("eagle3", "8b"): (
            root
            / "reports/final/eagle3-8b-math500/math500-confirmatory-paper-summary.json",
            "relay_eagle3",
        ),
        ("eagle3", "14b"): (
            root
            / "reports/final/eagle3-14b-math500/math500-confirmatory-paper-summary.json",
            "relay_eagle3",
        ),
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(
        development, key=lambda item: (FAMILY_ORDER[item[0]], TARGET_ORDER[item[1]])
    ):
        path, keys = development[key]
        value: Any = _read_json(path)
        for item in keys:
            value = value[item]
        summary_path, method_name = heldout[key]
        method = _read_json(summary_path)["methods"][method_name]
        observed = float(method["end_to_end_speedup_vs_reference"])
        estimate = float(value)
        rows.append(
            {
                "family_key": key[0],
                "family": FAMILY_LABEL[key[0]],
                "target_key": key[1],
                "target": TARGET_LABEL[key[1]],
                "development_estimate": estimate,
                "heldout_speedup": observed,
                "heldout_ci": [
                    float(number)
                    for number in method["end_to_end_speedup_vs_reference_ci95"]
                ],
                "relative_error_percent": 100.0 * (observed / estimate - 1.0),
            }
        )
    return rows


def load_paper_data(root: Path) -> dict[str, Any]:
    root = root.resolve()
    breadth_rows, breadth_aggregate = _load_breadth(root)
    memory = {
        "8b": _read_json(root / "reports/final/EAGLE3_8B_MEMORY.json"),
        "14b": _read_json(root / "reports/final/EAGLE3_14B_MEMORY.json"),
    }
    forecasts = _load_heldout_forecasts(root)
    return {
        "main_math": _load_main_math(root),
        "paired_ar": _read_json(root / "reports/final/MAIN_PAIRED_AR.json"),
        "transfer_ar": _read_json(root / "reports/final/TRANSFER_PAIRED_AR.json"),
        "breadth_rows": breadth_rows,
        "breadth_aggregate": breadth_aggregate,
        "memory": memory,
        "training_seconds": _load_training(root),
        "native_controls": _load_native_controls(root),
        "overlap": _read_json(root / "reports/final/PROMPT_OVERLAP_AUDIT.json"),
        "similarity": _read_json(root / "reports/final/PROMPT_SIMILARITY_AUDIT.json"),
        "heldout_forecasts": forecasts,
        "forecast_mean_absolute_relative_error_percent": sum(
            abs(row["relative_error_percent"]) for row in forecasts
        )
        / len(forecasts),
    }


def _macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def _paired_ar_macros(data: dict[str, Any]) -> list[str]:
    """Macros for every headline number, so prose can never drift from the tables."""
    name = {
        ("dflash", "8b"): "PairedDFlashEight",
        ("dflash", "14b"): "PairedDFlashFourteen",
        ("eagle3", "8b"): "PairedEagleEight",
        ("eagle3", "14b"): "PairedEagleFourteen",
    }
    lines: list[str] = []
    recoveries: list[float] = []
    for pair in data["paired_ar"]["pairs"]:
        key = (pair["family"], pair["target"])
        relay = pair["relay_vs_ar"]["estimate"]
        lines.extend(
            [
                _macro(f"{name[key]}Ar", f"{relay:.2f}"),
                _macro(f"{name[key]}ArLow", f"{pair['relay_vs_ar']['lower']:.2f}"),
                _macro(f"{name[key]}ArHigh", f"{pair['relay_vs_ar']['upper']:.2f}"),
            ]
        )
        specific = pair["target_specific_vs_ar"]
        if specific is not None:
            recovery = 100.0 * relay / specific["estimate"]
            recoveries.append(recovery)
            lines.append(_macro(f"{name[key]}Recovery", f"{recovery:.1f}"))
    lines.extend(
        [
            _macro("RecoveryLow", f"{min(recoveries):.0f}"),
            _macro("RecoveryHigh", f"{max(recoveries):.0f}"),
        ]
    )
    transfer = {setting["key"]: setting for setting in data["transfer_ar"]["settings"]}
    for key, macro in (
        ("fine_tuned_descendant", "TransferDescendant"),
        ("adapter_small_to_large", "TransferAdapter"),
        ("cross_family_smaller_target", "TransferCrossFamily"),
        ("cross_tokenizer", "TransferCrossTokenizer"),
    ):
        lines.append(_macro(macro, f"{transfer[key]['relay_vs_ar']['estimate']:.2f}"))
    return lines


def _write_macros(data: dict[str, Any], path: Path) -> None:
    rows = {(row["family_key"], row["target_key"]): row for row in data["main_math"]}
    prefix = {
        ("dflash", "8b"): "DFlashEight",
        ("dflash", "14b"): "DFlashFourteen",
        ("eagle3", "8b"): "EagleEight",
        ("eagle3", "14b"): "EagleFourteen",
    }
    lines = ["% Generated by scripts/build_iclr_paper_assets.py. Do not edit."]
    for key in sorted(
        prefix, key=lambda item: (FAMILY_ORDER[item[0]], TARGET_ORDER[item[1]])
    ):
        row = rows[key]
        name = prefix[key]
        lines.extend(
            [
                _macro(f"{name}SourceTps", f"{row['source_tps']:.1f}"),
                _macro(f"{name}RelayTps", f"{row['relay_tps']:.1f}"),
                _macro(f"{name}Speedup", f"{row['speedup']:.3f}"),
                _macro(f"{name}SpeedupLow", f"{row['speedup_ci'][0]:.3f}"),
                _macro(f"{name}SpeedupHigh", f"{row['speedup_ci'][1]:.3f}"),
                _macro(f"{name}Accuracy", f"{100.0 * row['relay_accuracy']:.1f}"),
                _macro(f"{name}Exact", f"{row['exact_matches']}/{row['requests']}"),
                _macro(
                    f"{name}SourceShare", f"{100.0 * row['source_trunk_fraction']:.1f}"
                ),
                _macro(
                    f"{name}AcceptanceRetention",
                    f"{100.0 * row['acceptance_retention']:.1f}",
                ),
            ]
        )
    aggregate = data["breadth_aggregate"]
    lines.extend(
        [
            _macro("BreadthCells", str(aggregate["benchmark_cells"])),
            _macro("BreadthPositiveCells", str(aggregate["positive_speed_ci_cells"])),
            _macro(
                "BreadthDirectionMatches", str(aggregate["amdahl_direction_matches"])
            ),
            _macro(
                "BreadthAccountingError",
                f"{100.0 * aggregate['amdahl_mean_absolute_relative_error']:.2f}",
            ),
            _macro(
                "BreadthRawGeomean", f"{aggregate['raw_geometric_mean_speedup']:.3f}"
            ),
            _macro(
                "BreadthPolicyGeomean",
                f"{aggregate['profile_policy_geometric_mean_speedup']:.3f}",
            ),
            _macro("FitExamples", "4,096"),
            _macro("FitSteps", "1,024"),
            _macro("RelayParamsEight", "52.4M"),
            _macro("RelayParamsFourteen", "65.5M"),
            _macro(
                "HeldoutForecastError",
                f"{data['forecast_mean_absolute_relative_error_percent']:.2f}",
            ),
        ]
    )
    for target, label in (("8b", "Eight"), ("14b", "Fourteen")):
        mem = data["memory"][target]
        lines.extend(
            [
                _macro(
                    f"MemoryPeakSave{label}", f"{mem['peak_allocated_saved_gib']:.2f}"
                ),
                _macro(
                    f"MemoryPeakSavePercent{label}",
                    f"{100.0 * mem['peak_allocated_saved_fraction']:.1f}",
                ),
                _macro(
                    f"MemorySteadySave{label}",
                    f"{mem['allocated_before_saved_gib']:.2f}",
                ),
            ]
        )
    lines.extend(_paired_ar_macros(data))
    training = data["training_seconds"]
    lines.extend(
        [
            _macro("FastestFitSeconds", f"{min(training.values()):.1f}"),
            _macro("SlowestFitSeconds", f"{max(training.values()):.1f}"),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_main_table(data: dict[str, Any], path: Path) -> None:
    """Main MATH-500 table, every column from one paired run per pair.

    Speedups are all against plain autoregressive decoding, which is
    measured inside the same run as the method it is compared against, so
    no row mixes references or executions.
    """
    lines = [
        "% Generated by scripts/build_iclr_paper_assets.py. Do not edit.",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{Speedup over plain AR} & Relay over & Accepted tokens \\\\",
        "Proposer & Target & Source reuse & \\method{} [95\\% CI] & source reuse & per cycle S/R \\\\",
        "\\midrule",
    ]
    for pair in data["paired_ar"]["pairs"]:
        relay = pair["relay_vs_ar"]
        lines.append(
            f"{FAMILY_LABEL[pair['family']]} & {TARGET_LABEL[pair['target']]} & "
            f"{pair['source_reuse_vs_ar']['estimate']:.2f}x & "
            f"\\textbf{{{relay['estimate']:.2f}x}} "
            f"[{relay['lower']:.2f}, {relay['upper']:.2f}] & "
            f"{pair['relay_vs_source_reuse']['estimate']:.2f}x & "
            f"{pair['accept_source_reuse']:.2f}/{pair['accept_relay']:.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_transfer_table(data: dict[str, Any], path: Path) -> None:
    """Transfer settings, same paired-against-plain-AR statistic as the main table."""
    lines = [
        "% Generated by scripts/build_iclr_paper_assets.py. Do not edit.",
        "\\begin{tabular}{p{0.52\\linewidth}rr}",
        "\\toprule",
        "Change to the target & \\method{} over plain AR & Accepted tokens \\\\",
        " & [95\\% CI] & per cycle \\\\",
        "\\midrule",
    ]
    for setting in data["transfer_ar"]["settings"]:
        relay = setting["relay_vs_ar"]
        lines.append(
            f"{setting['label']} & "
            f"\\textbf{{{relay['estimate']:.2f}x}} "
            f"[{relay['lower']:.2f}, {relay['upper']:.2f}] & "
            f"{setting['accept_relay']:.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_acceptance_table(data: dict[str, Any], path: Path) -> None:
    lines = [
        "% Generated by scripts/build_iclr_paper_assets.py. Do not edit.",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Proposer & Target & Source accept. & Relay accept. & Retained & Removed time \\\\",
        " & & \\multicolumn{2}{c}{tokens/cycle} & \\multicolumn{2}{c}{\\%} \\\\",
        "\\midrule",
    ]
    # Accepted-token counts come from MAIN_PAIRED_AR.json, the same artifact
    # Table 2 is generated from, so the two tables cannot report different
    # values for the same quantity.
    paired_ar = {
        (pair["family"], pair["target"]): pair for pair in data["paired_ar"]["pairs"]
    }
    for row in data["main_math"]:
        pair = paired_ar[(row["family_key"], row["target_key"])]
        source_accept = pair["accept_source_reuse"]
        relay_accept = pair["accept_relay"]
        lines.append(
            f"{row['family']} & {row['target']} & {source_accept:.2f} & "
            f"{relay_accept:.2f} & {100.0 * relay_accept / source_accept:.1f} & "
            f"{100.0 * row['source_trunk_fraction']:.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_breadth_table(data: dict[str, Any], path: Path) -> None:
    lines = [
        "% Generated by scripts/build_iclr_paper_assets.py. Do not edit.",
        "\\begin{tabular}{lllrrrrr}",
        "\\toprule",
        "Proposer & Target & Task & $n$/clusters & Source & Relay & Speedup [95\\% CI] & Exact \\\\",
        " & & & & \\multicolumn{2}{c}{output tok/s} & & {\\%} \\\\",
        "\\midrule",
    ]
    previous: tuple[str, str] | None = None
    for row in data["breadth_rows"]:
        current = (row["family"], row["target"])
        if previous is not None and current != previous:
            lines.append("\\addlinespace[1.5pt]")
        lines.append(
            f"{row['family']} & {row['target']} & {row['task']} & "
            f"{row['requests']}/{row['clusters']} & {row['source_tps']:.1f} & "
            f"{row['relay_tps']:.1f} & {row['speedup']:.3f} "
            f"[{row['speedup_ci'][0]:.3f}, {row['speedup_ci'][1]:.3f}] & "
            f"{100.0 * row['exact_rate']:.1f} \\\\"
        )
        previous = current
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_resource_table(data: dict[str, Any], path: Path) -> None:
    """Deployment cost of one map. Every cell is defined for every row.

    The target-specific-proposer recovery percentages deliberately live in
    prose rather than here, because no public DFlash checkpoint exists for
    a 14B target and a table cell for it could only be left empty.
    """
    lines = [
        "% Generated by scripts/build_iclr_paper_assets.py. Do not edit.",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Target & Relay parameters & Fit time, DFlash/EAGLE-3 & Peak GPU memory saved \\\\",
        " & {millions} & {seconds} & {GiB / \\%} \\\\",
        "\\midrule",
    ]
    for target in ("8b", "14b"):
        params = 52.4288 if target == "8b" else 65.536
        mem = data["memory"][target]
        fit_d = data["training_seconds"][("dflash", target)]
        fit_e = data["training_seconds"][("eagle3", target)]
        lines.append(
            f"{TARGET_LABEL[target]} & {params:.1f} & {fit_d:.1f}/{fit_e:.1f} & "
            f"{mem['peak_allocated_saved_gib']:.2f} / "
            f"{100.0 * mem['peak_allocated_saved_fraction']:.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_forecast_table(data: dict[str, Any], path: Path) -> None:
    lines = [
        "% Generated by scripts/build_iclr_paper_assets.py. Do not edit.",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Proposer & Target & Development estimate & Held-out speedup & Relative difference \\\\",
        " & & & {95\\% CI} & {\\%} \\\\",
        "\\midrule",
    ]
    for row in data["heldout_forecasts"]:
        lines.append(
            f"{row['family']} & {row['target']} & {row['development_estimate']:.3f} & "
            f"{row['heldout_speedup']:.3f} "
            f"[{row['heldout_ci'][0]:.3f}, {row['heldout_ci'][1]:.3f}] & "
            f"{row['relative_error_percent']:+.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_similarity_table(data: dict[str, Any], path: Path) -> None:
    labels = {
        "math500": "MATH-500",
        "gsm8k": "GSM8K",
        "humaneval": "HumanEval",
        "mbpp": "MBPP",
        "mtbench": "MT-Bench",
    }
    rows = data["similarity"]["by_benchmark"]
    lines = [
        "% Generated by scripts/build_iclr_paper_assets.py. Do not edit.",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Task & Maximum & $\\geq0.80$ & $\\geq0.90$ & $\\geq0.95$ \\\\",
        "\\midrule",
    ]
    for task in SIMILARITY_TASK_ORDER:
        row = rows[task]
        counts = row["counts"]
        lines.append(
            f"{labels[task]} & {row['maximum']:.3f} & {counts['0.80']} & "
            f"{counts['0.90']} & {counts['0.95']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_pdf(fig: Any, path: Path) -> None:
    fig.savefig(path, format="pdf", bbox_inches="tight", metadata=PDF_METADATA)
    plt.close(fig)


def _box(
    ax: Any, x: float, y: float, w: float, h: float, text: str, **kwargs: Any
) -> None:
    facecolor = kwargs.pop("facecolor", "#f5f5f5")
    edgecolor = kwargs.pop("edgecolor", "#444444")
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.9,
        **kwargs,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.0)


def _arrow(
    ax: Any, start: tuple[float, float], end: tuple[float, float], **kwargs: Any
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=0.9,
        color=kwargs.pop("color", "#333333"),
        connectionstyle=kwargs.pop("connectionstyle", "arc3"),
        **kwargs,
    )
    ax.add_patch(arrow)


def _orthogonal_link(
    ax: Any,
    start: tuple[float, float],
    corner: tuple[float, float],
    end: tuple[float, float],
) -> None:
    """A two-bend, purely right-angle connector from one box edge to another.

    Both endpoints sit exactly on a box boundary and every segment is
    axis-aligned, so there is no diagonal that can graze a corner and no
    leg that appears to begin in empty space. This matches the routing
    convention used by this paper's TikZ diagrams.
    """
    color = "#333333"
    corner_x, corner_y = corner
    ax.plot(
        [start[0], corner_x],
        [start[1], corner_y],
        color=color,
        linewidth=0.9,
        zorder=2,
        solid_capstyle="projecting",
    )
    arrow = FancyArrowPatch(
        (corner_x, corner_y),
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=0.9,
        color=color,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arrow)


def _draw_system_overview(path: Path) -> None:
    _set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    for ax, title in zip(axes, ("Source reuse", "RelaySpec"), strict=True):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontweight="bold", pad=2)

    left, right = axes
    _box(left, 0.04, 0.58, 0.28, 0.18, "Qwen3-4B\nlayers 0 to 33", facecolor="#d9d9d9")
    _box(left, 0.39, 0.58, 0.25, 0.18, "Expected\ncontext")
    _box(left, 0.71, 0.58, 0.25, 0.18, "Frozen\nproposer", facecolor="#dbe9f6")
    _box(left, 0.38, 0.12, 0.30, 0.19, "Full target\nverifier", facecolor="#dff0d8")
    _arrow(left, (0.32, 0.67), (0.39, 0.67))
    _arrow(left, (0.64, 0.67), (0.71, 0.67))
    # Proposer down and back into the verifier's right edge, then verifier
    # out of its left edge and back up into the first box. Both loops are
    # fully orthogonal and stay on opposite sides of the figure.
    _orthogonal_link(left, (0.835, 0.58), (0.835, 0.215), (0.68, 0.215))
    _orthogonal_link(left, (0.38, 0.215), (0.18, 0.215), (0.18, 0.58))
    left.text(
        0.18,
        0.82,
        "run after each committed block",
        ha="center",
        fontsize=6.8,
        color="#444444",
    )
    left.text(
        0.53, 0.04, "Only the full target accepts tokens", ha="center", fontsize=7.5
    )

    _box(
        right, 0.04, 0.58, 0.28, 0.18, "Five target\nhidden states", facecolor="#dff0d8"
    )
    _box(right, 0.39, 0.58, 0.25, 0.18, "Linear\nrelay", facecolor="#fff2cc")
    _box(right, 0.71, 0.58, 0.25, 0.18, "Frozen\nproposer", facecolor="#dbe9f6")
    _box(right, 0.38, 0.12, 0.30, 0.19, "Full target\nverifier", facecolor="#dff0d8")
    _arrow(right, (0.32, 0.67), (0.39, 0.67))
    _arrow(right, (0.64, 0.67), (0.71, 0.67))
    # Proposer down and back into the verifier's right edge, then verifier
    # out of its left edge and back up into the first box. Both loops are
    # fully orthogonal and stay on opposite sides of the figure.
    _orthogonal_link(right, (0.835, 0.58), (0.835, 0.215), (0.68, 0.215))
    _orthogonal_link(right, (0.38, 0.215), (0.18, 0.215), (0.18, 0.58))
    right.text(0.53, 0.04, "The 4B layers are not loaded", ha="center", fontsize=7.5)
    fig.subplots_adjust(wspace=0.12)
    _save_pdf(fig, path)


def _draw_main_throughput(data: dict[str, Any], path: Path) -> None:
    _set_plot_style()
    rows = data["main_math"]
    controls = data["native_controls"]
    paired_ar = {
        (pair["family"], pair["target"]): pair for pair in data["paired_ar"]["pairs"]
    }
    y = list(range(len(rows)))[::-1]
    fig, ax = plt.subplots(figsize=(6.55, 2.45))

    colors = {
        "ar": "#777777",
        "source": "#8f8f8f",
        "relay": "#0072B2",
        "native": "#D89000",
    }
    for yi, row in zip(y, rows, strict=True):
        key = (row["family_key"], row["target_key"])
        control = controls[key]
        source = row["source_tps"]
        relay = row["relay_tps"]
        ax.plot([source, relay], [yi, yi], color="#0072B2", linewidth=2.2, zorder=1)
        ax.scatter(
            float(control["native_ar_tps"]),
            yi,
            marker="D",
            s=27,
            color=colors["ar"],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        ax.scatter(
            source,
            yi,
            marker="o",
            s=34,
            facecolor="white",
            edgecolor=colors["source"],
            linewidth=1.1,
            zorder=3,
        )
        ax.scatter(
            relay,
            yi,
            marker="o",
            s=38,
            color=colors["relay"],
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        native = control["native_proposer_tps"]
        if native is not None:
            ax.scatter(
                float(native),
                yi,
                marker="^",
                s=38,
                color=colors["native"],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        # Annotate with the SAME statistic Table 2 reports: the paired
        # bootstrap speedup over plain autoregressive decoding. Using any
        # other ratio here makes the figure contradict the table.
        paired = paired_ar[key]["relay_vs_ar"]
        midpoint = (source + relay) / 2.0
        ax.text(
            midpoint,
            yi + 0.16,
            f"{paired['estimate']:.2f}x vs. AR"
            f"  [{paired['lower']:.2f}, {paired['upper']:.2f}]",
            ha="center",
            va="bottom",
            fontsize=6.6,
            color="#005f94",
        )

    ax.set_yticks(y)
    ax.set_yticklabels([f"{row['family']}, Qwen3-{row['target']}" for row in rows])
    ax.set_xlabel("End-to-end output tokens per second")
    ax.set_xlim(0, 265)
    ax.set_ylim(-0.55, len(rows) - 0.25)
    ax.grid(axis="x", color="#dddddd", linewidth=0.5, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [
        ax.scatter(
            [], [], marker="D", s=27, color=colors["ar"], label="Autoregressive"
        ),
        ax.scatter(
            [],
            [],
            marker="o",
            s=34,
            facecolor="white",
            edgecolor=colors["source"],
            label="Source reuse",
        ),
        ax.scatter([], [], marker="o", s=38, color=colors["relay"], label="RelaySpec"),
        ax.scatter(
            [],
            [],
            marker="^",
            s=38,
            color=colors["native"],
            label="Target-specific proposer",
        ),
    ]
    ax.legend(
        handles=handles,
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        columnspacing=1.2,
        handletextpad=0.4,
    )
    _save_pdf(fig, path)


def _draw_mechanism(data: dict[str, Any], path: Path) -> None:
    _set_plot_style()
    rows = data["main_math"]
    labels = [f"{row['family']}\n{row['target']}" for row in rows]
    categories = [
        ("Target", "target_fraction", "#4c78a8", ""),
        ("Source trunk", "source_trunk_fraction", "#b3b3b3", "///"),
        ("Proposer", "proposer_fraction", "#f2cf5b", ".."),
        ("Other", "other_fraction", "#ffffff", "xx"),
    ]
    fig, (ax, gain) = plt.subplots(
        1, 2, figsize=(7.0, 2.65), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    bottoms = [0.0] * len(rows)
    for label, key, color, hatch in categories:
        values = [100.0 * row[key] for row in rows]
        ax.bar(
            labels,
            values,
            bottom=bottoms,
            label=label,
            color=color,
            edgecolor="#333333",
            linewidth=0.4,
            hatch=hatch,
        )
        bottoms = [
            bottom + value for bottom, value in zip(bottoms, values, strict=True)
        ]
    for index, row in enumerate(rows):
        target = 100.0 * row["target_fraction"]
        source = 100.0 * row["source_trunk_fraction"]
        ax.text(
            index,
            target + source / 2,
            f"{source:.1f}%",
            ha="center",
            va="center",
            fontsize=7,
        )
    ax.set_ylabel("Source-reuse request time (%)")
    ax.set_ylim(0, 104)
    ax.set_title("(a) Removable work")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.25), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)

    y = list(range(len(rows)))[::-1]
    for yi, row in zip(y, rows, strict=True):
        ideal = row["ideal_speedup"]
        measured = row["speedup"]
        gain.plot([measured, ideal], [yi, yi], color="#b0b0b0", linewidth=2.0)
        gain.scatter(
            ideal,
            yi,
            marker="o",
            s=30,
            facecolor="white",
            edgecolor="#777777",
            linewidth=1.0,
            zorder=3,
        )
        gain.scatter(
            measured,
            yi,
            marker="o",
            s=34,
            color="#0072B2",
            edgecolor="white",
            linewidth=0.4,
            zorder=4,
        )
        gain.text(
            measured,
            yi + 0.16,
            f"{100.0 * row['acceptance_retention']:.1f}% accepted-token retention",
            ha="left",
            va="bottom",
            fontsize=6.2,
            color="#005f94",
        )
    gain.axvline(1.0, color="#333333", linewidth=0.8)
    gain.set_yticks(y)
    gain.set_yticklabels(labels)
    gain.set_xlim(1.0, 1.69)
    gain.set_ylim(-0.55, len(rows) - 0.25)
    gain.set_xlabel("Relay/source throughput")
    gain.set_title("(b) Available and measured gain")
    gain.grid(axis="x", color="#dddddd", linewidth=0.5)
    gain.spines[["top", "right", "left"]].set_visible(False)
    gain.tick_params(axis="y", length=0)
    fig.subplots_adjust(wspace=0.38)
    _save_pdf(fig, path)


def _draw_breadth_and_margin(data: dict[str, Any], path: Path) -> None:
    _set_plot_style()
    rows = data["breadth_rows"]
    fig, (heat, scatter) = plt.subplots(
        1, 2, figsize=(7.0, 2.85), gridspec_kw={"width_ratios": [1.08, 1]}
    )

    row_keys = [
        ("dflash", "8b"),
        ("dflash", "14b"),
        ("eagle3", "8b"),
        ("eagle3", "14b"),
    ]
    task_keys = ["gsm8k", "humaneval", "mbpp", "mtbench"]
    by_key = {
        (row["family_key"], row["target_key"], row["task_key"]): row for row in rows
    }
    matrix = [
        [by_key[(family, target, task)]["speedup"] for task in task_keys]
        for family, target in row_keys
    ]
    image = heat.imshow(
        matrix,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=0.86, vcenter=1.0, vmax=1.44),
        aspect="auto",
    )
    for row_index, (family, target) in enumerate(row_keys):
        for column_index, task in enumerate(task_keys):
            item = by_key[(family, target, task)]
            low, high = item["speedup_ci"]
            if low > 1.0:
                edge = "#143f5c"
                width = 1.8
                linestyle = "solid"
            elif high < 1.0:
                edge = "#7a1515"
                width = 1.8
                linestyle = "dashed"
            else:
                edge = "#666666"
                width = 0.8
                linestyle = "dotted"
            heat.add_patch(
                plt.Rectangle(
                    (column_index - 0.5, row_index - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=edge,
                    linewidth=width,
                    linestyle=linestyle,
                )
            )
            heat.text(
                column_index,
                row_index,
                f"{item['speedup']:.3f}x",
                ha="center",
                va="center",
                fontsize=7.0,
                color="white"
                if item["speedup"] >= 1.28 or item["speedup"] <= 0.91
                else "#222222",
            )
    heat.set_xticks(range(len(task_keys)))
    heat.set_xticklabels(
        [TASK_LABEL[key] for key in task_keys], rotation=28, ha="right"
    )
    heat.set_yticks(range(len(row_keys)))
    heat.set_yticklabels(
        [
            f"{FAMILY_LABEL[family]}, {TARGET_LABEL[target]}"
            for family, target in row_keys
        ]
    )
    heat.set_title("(a) Throughput across tasks")
    heat.tick_params(length=0)
    colorbar = fig.colorbar(image, ax=heat, fraction=0.045, pad=0.035)
    colorbar.set_label("Relay/source")
    colorbar.ax.tick_params(labelsize=6.5)

    marker = {"dflash": "o", "eagle3": "s"}
    color = {"8b": "#0072B2", "14b": "#D55E00"}
    for row in rows:
        scatter.scatter(
            100.0 * row["acceptance_margin"],
            row["speedup"],
            marker=marker[row["family_key"]],
            s=28,
            facecolor=(
                color[row["target_key"]] if row["target_key"] == "8b" else "white"
            ),
            edgecolor=color[row["target_key"]],
            linewidth=0.8,
            zorder=3,
        )
    scatter.axvline(0.0, color="#333333", linewidth=0.8)
    scatter.axhline(1.0, color="#333333", linewidth=0.8)
    scatter.set_xlim(-10, 29)
    scatter.set_ylim(0.85, 1.47)
    scatter.set_xlabel("Acceptance margin above break-even (pp)")
    scatter.set_ylabel("Measured relay/source throughput")
    scatter.set_title("(b) Measured break-even boundary")
    scatter.grid(color="#e0e0e0", linewidth=0.5)
    scatter.spines[["top", "right"]].set_visible(False)

    handles = []
    labels = []
    for family in ("dflash", "eagle3"):
        handle = scatter.scatter([], [], marker=marker[family], color="#777777", s=24)
        handles.append(handle)
        labels.append(FAMILY_LABEL[family])
    for target in ("8b", "14b"):
        handle = scatter.scatter(
            [],
            [],
            marker="o",
            facecolor=color[target] if target == "8b" else "white",
            edgecolor=color[target],
            s=24,
        )
        handles.append(handle)
        labels.append(TARGET_LABEL[target])
    scatter.legend(handles, labels, frameon=False, ncol=2, loc="upper left")
    fig.subplots_adjust(wspace=0.40)
    _save_pdf(fig, path)


def _draw_memory(data: dict[str, Any], path: Path) -> None:
    _set_plot_style()
    targets = ("8b", "14b")
    gib = float(1024**3)
    source = [
        float(data["memory"][target]["source_peak_allocated_bytes"]) / gib
        for target in targets
    ]
    relay = [
        float(data["memory"][target]["relay_peak_allocated_bytes"]) / gib
        for target in targets
    ]
    positions = [0.0, 1.0]
    width = 0.30
    fig, ax = plt.subplots(figsize=(4.65, 2.35))
    ax.bar(
        [position - width / 2 for position in positions],
        source,
        width,
        label="Source reuse",
        color="#b3b3b3",
        edgecolor="#333333",
        linewidth=0.5,
    )
    ax.bar(
        [position + width / 2 for position in positions],
        relay,
        width,
        label="RelaySpec",
        color="#0072B2",
        edgecolor="#333333",
        linewidth=0.5,
    )
    for index, target in enumerate(targets):
        item = data["memory"][target]
        ax.text(
            positions[index],
            max(source[index], relay[index]) + 1.1,
            f"-{item['peak_allocated_saved_gib']:.2f} GiB "
            f"(-{100.0 * item['peak_allocated_saved_fraction']:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="#005f94",
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(["Qwen3-8B", "Qwen3-14B"])
    ax.set_ylabel("Peak allocated GPU memory (GiB)")
    ax.set_ylim(0, 43)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    _save_pdf(fig, path)


def _draw_acceptance_survival(data: dict[str, Any], path: Path) -> None:
    _set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.3), sharey=True)
    for ax, row in zip(axes.flat, data["main_math"], strict=True):
        positions = sorted(row["source_survival"])
        source = [100.0 * row["source_survival"][position] for position in positions]
        relay = [100.0 * row["relay_survival"][position] for position in positions]
        ax.plot(
            positions,
            source,
            color="#444444",
            marker="o",
            markersize=2.6,
            label="Source",
        )
        ax.plot(
            positions, relay, color="#0072B2", marker="s", markersize=2.6, label="Relay"
        )
        ax.set_title(f"{row['family']}, target {row['target']}")
        ax.set_xlabel("Draft position")
        ax.set_xlim(min(positions), max(positions))
        ax.set_ylim(0, 103)
        ax.grid(color="#e0e0e0", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Accepted prefix survives (%)")
    axes[1, 0].set_ylabel("Accepted prefix survives (%)")
    axes[0, 1].legend(frameon=False, loc="upper right")
    fig.subplots_adjust(hspace=0.42, wspace=0.20)
    _save_pdf(fig, path)


def build_all(root: Path, output: Path) -> dict[str, Any]:
    from relayspec.ar_paper_evidence import build_ar_assets
    from relayspec.paper_evidence import build_current_assets

    data = build_current_assets(root, output, load_paper_data(root))
    data["ar_revision"] = build_ar_assets(root, output)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="RelaySpec repository root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Paper directory. Defaults to paper/iclr2027 under the repository root.",
    )
    args = parser.parse_args()
    output = args.output or args.root / "paper" / "iclr2027"
    data = build_all(args.root, output)
    print(
        f"Generated {len(data['main_math'])} main cells and "
        f"{len(data['breadth_rows'])} breadth cells in {output}"
    )


if __name__ == "__main__":
    main()
