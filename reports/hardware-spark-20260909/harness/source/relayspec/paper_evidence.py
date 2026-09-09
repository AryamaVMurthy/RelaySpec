"""Build the current manuscript from the recorded source-reuse experiments.

The evidence boundary deliberately excludes exploratory plain-target and transfer
registries. All complete workload cells are retained, regardless of direction.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

BLUE = "#355C7D"
GREEN = "#277D73"
ORANGE = "#AF6B36"
META = {
    "CreationDate": dt.datetime(2026, 9, 5, tzinfo=dt.UTC),
    "ModDate": dt.datetime(2026, 9, 5, tzinfo=dt.UTC),
    "Creator": "RelaySpec recorded-evidence builder",
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _save(fig: Any, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", metadata=META)
    plt.close(fig)


def _table(path: Path, columns: str, header: str, rows: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                r"\begin{tabular}{" + columns + "}",
                r"\toprule",
                header + r" \\",
                r"\midrule",
                *[r + r" \\" for r in rows],
                r"\bottomrule",
                r"\end{tabular}",
                "",
            ]
        )
    )


def verify_recorded_pairs(
    root: Path, row: dict[str, Any], directory: Path, benchmark: str
) -> dict[str, Any]:
    """Check raw/scored row identity, completeness, times, quality and hashes."""
    raw_files = sorted(directory.glob("benchmark-rank*.jsonl"))
    scored = directory / "benchmark-scored.jsonl"
    raw = [json.loads(line) for p in raw_files for line in p.open() if line.strip()]
    raw = [x for x in raw if x["benchmark"] == benchmark]
    if not raw:
        raise ValueError(f"No raw evidence for {directory}/{benchmark}")
    groups: dict[tuple[str, int], dict[str, Any]] = {}
    for x in raw:
        key = (str(x["problem_id"]), int(x["repetition"]))
        group = groups.setdefault(key, {})
        if x["method"] in group:
            raise ValueError(f"Duplicate raw row: {directory}/{key}")
        if x["request_seconds"] <= 0:
            raise ValueError("Non-positive request time")
        group[x["method"]] = x
    source = (
        "optimized_source_reuse"
        if row["family_key"] == "dflash"
        else "source_reuse_eagle3"
    )
    relay = "relay_p" if row["family_key"] == "dflash" else "relay_eagle3"
    if len(groups) != row["requests"] or any(
        set(v) != {source, relay} for v in groups.values()
    ):
        raise ValueError(f"Incomplete measurement: {directory}/{benchmark}")
    throughput = {
        m: sum(v[m]["output_tokens"] for v in groups.values())
        / sum(v[m]["request_seconds"] for v in groups.values())
        for m in [source, relay]
    }
    for m, key in [(source, "source_tps"), (relay, "relay_tps")]:
        if not math.isclose(throughput[m], row[key], rel_tol=1e-10):
            raise ValueError(f"Stale throughput: {directory}/{benchmark}/{m}")
    if not math.isclose(
        throughput[relay] / throughput[source], row["speedup"], rel_tol=1e-10
    ):
        raise ValueError("Stale speed ratio")
    exact = sum(
        v[source]["output_hash"] == v[relay]["output_hash"] for v in groups.values()
    )
    if not math.isclose(exact / len(groups), row["exact_rate"], abs_tol=1e-12):
        raise ValueError("Stale token agreement")
    result = {
        "directory": str(directory.relative_to(root)),
        "benchmark": benchmark,
        "paired_requests": len(groups),
        "token_matches": exact,
        "source_tps": throughput[source],
        "relay_tps": throughput[relay],
        "source_mean_output_tokens": sum(
            v[source]["output_tokens"] for v in groups.values()
        )
        / len(groups),
        "relay_mean_output_tokens": sum(
            v[relay]["output_tokens"] for v in groups.values()
        )
        / len(groups),
    }
    if benchmark in {"math500", "gsm8k"}:
        scores = [json.loads(line) for line in scored.open() if line.strip()]
        scores = [x for x in scores if x["benchmark"] == benchmark]
        if len(scores) != 2 * len(groups):
            raise ValueError("Incomplete scored evidence")
        seen = set()
        for x in scores:
            key = (str(x["problem_id"]), int(x["repetition"]))
            sk = (*key, x["method"])
            if sk in seen:
                raise ValueError("Duplicate score")
            seen.add(sk)
            original = groups[key][x["method"]]
            for field in (
                "output_hash",
                "completion",
                "output_tokens",
                "request_seconds",
            ):
                if x[field] != original[field]:
                    raise ValueError(f"Scored row does not match raw row: {field}")
        accuracy = {
            m: sum(bool(x["correct"]) for x in scores if x["method"] == m) / len(groups)
            for m in [source, relay]
        }
        if benchmark == "math500":
            assert accuracy[source] == row["source_accuracy"]
            assert accuracy[relay] == row["relay_accuracy"]
        else:
            assert accuracy[source] == row["official_quality"][source]["accuracy"]
            assert accuracy[relay] == row["official_quality"][relay]["accuracy"]
        by_id = {
            (x["problem_id"], x["repetition"], x["method"]): x["correct"]
            for x in scores
        }
        result["score_disagreements"] = sum(
            by_id[(*k, source)] != by_id[(*k, relay)] for k in groups
        )
        result["accuracy"] = accuracy
    score_files = []
    if benchmark in {"humaneval", "mbpp"}:
        summary_path = directory / "evalplus-summary.json"
        summary = _json(summary_path)
        score_files.append(summary_path)
        statuses = {}
        for method in [source, relay]:
            metric = summary["benchmarks"][benchmark][method]
            if metric != row["official_quality"][method]:
                raise ValueError("Code score registry does not match scorer summary")
            score_path = directory / "evalplus" / metric["result_file"]
            score_files.append(score_path)
            evaluated = _json(score_path)["eval"]
            if len(evaluated) != len(groups) or any(
                len(v) != 1 for v in evaluated.values()
            ):
                raise ValueError("Code results must contain one program per task")
            statuses[method] = {
                key: (
                    v[0]["base_status"] == "pass",
                    v[0]["base_status"] == "pass" and v[0]["plus_status"] == "pass",
                )
                for key, v in evaluated.items()
            }
            for index, score_key in [(0, "base_pass_at_1"), (1, "plus_pass_at_1")]:
                observed = sum(v[index] for v in statuses[method].values()) / len(
                    groups
                )
                if not math.isclose(observed, metric[score_key], abs_tol=1e-12):
                    raise ValueError("Code pass counts disagree with recorded summary")
        if set(statuses[source]) != set(statuses[relay]):
            raise ValueError("Code methods have different task IDs")
        result["score_disagreements"] = sum(
            statuses[source][key] != statuses[relay][key] for key in statuses[source]
        )
    result["sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*raw_files, *([scored] if scored.exists() else []), *score_files]
    }
    return result


def build_current_assets(
    root: Path, output: Path, data: dict[str, Any]
) -> dict[str, Any]:
    generated, figures = output / "generated", output / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    main, breadth = data["main_math"], data["breadth_rows"]
    evidence = []
    for row in main:
        evidence.append(
            verify_recorded_pairs(
                root, row, root / row["source_path"].rsplit("/", 1)[0], "math500"
            )
        )
    for row in breadth:
        family, target, task = row["family_key"], row["target_key"], row["task_key"]
        directory = root / "reports/final" / f"{family}-{target}-breadth"
        if family == "eagle3" and task == "mtbench":
            directory = root / "reports/final" / f"{family}-{target}-mtbench-two-turn"
        evidence.append(verify_recorded_pairs(root, row, directory, task))
    extra = [
        root / "reports/final/BREADTH_MATRIX.json",
        root / "reports/final/PROMPT_SIMILARITY_AUDIT.json",
    ]
    extra += [root / row["source_path"] for row in main]
    extra += list((root / "reports/final").glob("EAGLE3_*_MEMORY.json"))
    extra += [
        root
        / x["source_path"].replace(
            "benchmark-paper-summary.json", "math500-confirmatory-paper-summary.json"
        )
        for x in main
    ]
    extra += list((root / "reports/training").glob("*/relay-training-summary.json"))
    extra += list(
        (root / "reports/design-selection/eagle3").glob("*/relay-training-summary.json")
    )
    extra += [
        root / "reports/design-selection/dflash/objective-selection.json",
        root / "reports/design-selection/eagle3/ARCHITECTURE_SELECTION.md",
        root / "reports/design-selection/eagle3/normalized-linear-analysis.json",
        root / "reports/eagle3-8b-scale-validation/analysis.json",
    ]
    registry = {
        "scope": "Recorded source-reuse comparison with task-scored math/code outputs",
        "interpretation": "Verifies saved evidence consistency, not decoder equivalence or independent rescoring.",
        "complete_workload_cells": len(evidence),
        "records": evidence,
        "summary_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(extra)
        },
    }
    (generated / "evidence_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n"
    )

    _table(
        generated / "main_math_table.tex",
        "llrrrr",
        r"Drafter & Target & Source tok/s & Relay tok/s & Ratio [95\% interval] & Score S / R",
        [
            f"{x['family']} & {x['target']} & {x['source_tps']:.2f} & {x['relay_tps']:.2f} & "
            f"{x['speedup']:.3f} [{x['speedup_ci'][0]:.3f}, {x['speedup_ci'][1]:.3f}] & "
            f"{100 * x['source_accuracy']:.1f} / {100 * x['relay_accuracy']:.1f}"
            for x in main
        ],
    )
    _table(
        generated / "breadth_table.tex",
        "lllrrrr",
        r"Drafter & Target & Task & Requests & Source tok/s & Relay tok/s & Ratio [95\% interval]",
        [
            f"{x['family']} & {x['target']} & {x['task']} & {x['requests']} & {x['source_tps']:.2f} & "
            f"{x['relay_tps']:.2f} & {x['speedup']:.4f} [{x['speedup_ci'][0]:.4f}, {x['speedup_ci'][1]:.4f}]"
            for x in breadth
        ],
    )
    quality = []
    for x in main:
        items = {
            r["task_key"]: r
            for r in breadth
            if (r["family_key"], r["target_key"]) == (x["family_key"], x["target_key"])
        }
        name = "relay_p" if x["family_key"] == "dflash" else "relay_eagle3"
        values = [100 * items["gsm8k"]["official_quality"][name]["accuracy"]]
        for task in ["humaneval", "mbpp"]:
            q = items[task]["official_quality"][name]
            values += [100 * q["base_pass_at_1"], 100 * q["plus_pass_at_1"]]
        quality.append(
            f"{x['family']} & {x['target']} & " + " & ".join(f"{v:.2f}" for v in values)
        )
    _table(
        generated / "quality_table.tex",
        "llrrrrr",
        "Drafter & Target & GSM8K & HumanEval & HumanEval+ & MBPP & MBPP+",
        quality,
    )
    _table(
        generated / "agreement_table.tex",
        "llrrr",
        "Drafter & Target & Matching token sequences & Mean tokens S & Mean tokens R",
        [
            f"{x['family']} & {x['target']} & {x['exact_matches']}/500 & {e['source_mean_output_tokens']:.2f} & {e['relay_mean_output_tokens']:.2f}"
            for x, e in zip(main, evidence[:4])
        ],
    )
    _table(
        generated / "resource_table.tex",
        "llrr",
        "Drafter & Target & Trainable weights (millions) & Fitting loop (seconds)",
        [
            f"{x['family']} & {x['target']} & {52.4288 if x['target_key'] == '8b' else 65.536:.2f} & "
            f"{data['training_seconds'][(x['family_key'], x['target_key'])]:.1f}"
            for x in main
        ],
    )
    _table(
        generated / "similarity_table.tex",
        "lrrrr",
        r"Task & Maximum overlap & $\geq 0.80$ & $\geq 0.90$ & $\geq 0.95$",
        [
            f"{label} & {q['maximum']:.3f} & {q['counts']['0.80']} & {q['counts']['0.90']} & {q['counts']['0.95']}"
            for task, label in [
                ("math500", "MATH-500"),
                ("gsm8k", "GSM8K"),
                ("humaneval", "HumanEval"),
                ("mbpp", "MBPP"),
                ("mtbench", "MT-Bench"),
            ]
            for q in [data["similarity"]["by_benchmark"][task]]
        ],
    )
    confirmed = []
    for x in main:
        p = root / x["source_path"].replace(
            "benchmark-paper-summary.json", "math500-confirmatory-paper-summary.json"
        )
        q = _json(p)["methods"]
        n = "relay_p" if x["family_key"] == "dflash" else "relay_eagle3"
        v = q[n]
        ci = v["end_to_end_speedup_vs_reference_ci95"]
        confirmed.append(
            f"{x['family']} & {x['target']} & {v['end_to_end_speedup_vs_reference']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] & {100 * v['accuracy']:.2f}"
        )
    _table(
        generated / "complement_table.tex",
        "llrr",
        "Drafter & Target & Ratio [95\\% interval] & Score S = R",
        confirmed,
    )

    objective = _json(root / "reports/design-selection/dflash/objective-selection.json")
    objective_rows = []
    for target in ["8B", "14B"]:
        metrics = objective["targets"]["Qwen/Qwen3-" + target]
        for key, label in [
            ("relative", "Relative error"),
            ("historical_0.1", "Squared error + cosine"),
        ]:
            q = metrics[key]
            lo, hi = q["source_relative_speedup_95_percent"]
            objective_rows.append(
                f"{target} & {label} & {q['source_relative_speedup']:.3f} [{lo:.3f}, {hi:.3f}] & {q['exact_pairs']}/32"
            )
    _table(
        generated / "objective_table.tex",
        "llrr",
        r"Target & Objective & Throughput ratio [95\% interval] & Token matches",
        objective_rows,
    )
    design_rows = []
    design_md = (
        root / "reports/design-selection/eagle3/ARCHITECTURE_SELECTION.md"
    ).read_text()
    loss_by_label = {}
    for line in design_md.splitlines():
        if line.startswith("| normalized input") or line.startswith(
            "| scale preserving"
        ):
            cells = [v.strip() for v in line.split("|")]
            loss_by_label[cells[1]] = float(cells[2])
    for label, key, path in [
        (
            "Normalized features",
            "normalized input",
            "reports/design-selection/eagle3/normalized-linear-analysis.json",
        ),
        (
            "Raw features",
            "scale preserving",
            "reports/eagle3-8b-scale-validation/analysis.json",
        ),
    ]:
        q = _json(root / path)
        ci = q["paired_bootstrap_95_percent"]
        design_rows.append(
            f"{label} & {loss_by_label[key]:.3f} & {ci['estimate']:.3f} [{ci['lower']:.3f}, {ci['upper']:.3f}] & {100 * q['acceptance_retention']:.1f}\\%"
        )
    _table(
        generated / "scale_table.tex",
        "lrrr",
        r"Input to linear map & Relative error & Throughput ratio [95\% interval] & Progress retained",
        design_rows,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.6, 2.35),
        gridspec_kw={"width_ratios": [1.45, 1]},
        layout="constrained",
    )
    labels = [f"{x['family']} {x['target']}" for x in main]
    y = np.arange(4)
    axes[0].barh(
        y + 0.17,
        [x["source_tps"] for x in main],
        0.32,
        label="Source reuse",
        color=BLUE,
    )
    axes[0].barh(
        y - 0.17,
        [x["relay_tps"] for x in main],
        0.32,
        label="RelaySpec",
        color=GREEN,
    )
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Output tokens / second")
    axes[0].set_title("(a) Measured throughput", loc="left")
    axes[0].legend(loc="lower right", frameon=False)
    axes[1].axvline(1, color="0.5", linestyle="--", linewidth=1)
    for i, x in enumerate(main):
        v = x["speedup"]
        lo, hi = x["speedup_ci"]
        axes[1].errorbar(
            v, i, xerr=[[v - lo], [hi - v]], fmt="o", color=GREEN, capsize=3
        )
        axes[1].text(v + 0.025, i - 0.15, f"{v:.3f}", fontsize=8)
    axes[1].set_yticks(y, labels=[])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.98, 1.61)
    axes[1].set_xlabel("Relay / source throughput")
    axes[1].set_title("(b) Paired 95% intervals", loc="left")
    _save(fig, figures / "main_throughput.pdf")

    fig, ax = plt.subplots(figsize=(6.6, 3.4), layout="constrained")
    offsets = [-0.24, -0.08, 0.08, 0.24]
    markers = ["o", "s", "^", "D"]
    colors = [GREEN, BLUE, ORANGE, "#75507B"]
    for i, x in enumerate(main):
        rs = [
            r
            for r in breadth
            if (r["family_key"], r["target_key"]) == (x["family_key"], x["target_key"])
        ]
        for j, r in enumerate(rs):
            v = r["speedup"]
            lo, hi = r["speedup_ci"]
            ax.errorbar(
                v,
                j + offsets[i],
                xerr=[[v - lo], [hi - v]],
                fmt=markers[i],
                color=colors[i],
                capsize=2,
                label=f"{x['family']} {x['target']}" if j == 0 else None,
            )
    ax.axvline(1, color="0.45", linestyle="--", linewidth=1)
    ax.set_yticks(range(4), ["GSM8K", "HumanEval", "MBPP", "MT-Bench"])
    ax.invert_yaxis()
    ax.set_xlim(0.84, 1.5)
    ax.set_xlabel("Relay / source throughput (paired 95% intervals)")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    ax.grid(axis="x", alpha=0.15)
    _save(fig, figures / "breadth_and_margin.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), layout="constrained")
    shares = np.array(
        [
            [x[k] for x in main]
            for k in [
                "source_trunk_fraction",
                "target_fraction",
                "proposer_fraction",
                "other_fraction",
            ]
        ]
    )
    left = np.zeros(4)
    for a, c, component_label in zip(
        shares,
        [ORANGE, BLUE, GREEN, "#D6D9DB"],
        ["Source transformer", "Target", "Draft", "Other"],
    ):
        axes[0].barh(y, 100 * a, left=100 * left, color=c, label=component_label)
        left += a
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Source-reuse request time (%)")
    axes[0].legend(
        ncol=2,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.32),
        frameon=False,
    )
    for i, x in enumerate(main):
        a = sum(x["source_survival"].values())
        b = sum(x["relay_survival"].values())
        axes[1].plot([a, b], [i, i], color="0.7")
        axes[1].scatter(a, i, color=BLUE, marker="s")
        axes[1].scatter(b, i, color=GREEN, marker="o")
    axes[1].set_yticks(y, labels=[])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Recorded tokens advanced / cycle")
    axes[1].plot([], [], color=BLUE, marker="s", linestyle="", label="Source reuse")
    axes[1].plot([], [], color=GREEN, marker="o", linestyle="", label="RelaySpec")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False)
    _save(fig, figures / "mechanism.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.2), layout="constrained")
    for i, x in enumerate(main[::2]):
        ax = axes[i]
        for key, color, label, style in [
            ("source_survival", BLUE, "Source reuse", "-"),
            ("relay_survival", GREEN, "RelaySpec", "--"),
        ]:
            vals = x[key]
            ax.plot(
                sorted(vals),
                [vals[k] for k in sorted(vals)],
                style,
                color=color,
                label=label,
            )
        ax.set_ylim(0, 1.04)
        ax.set_title(f"{x['family']} / 8B")
        ax.set_xlabel("Position within recorded cycle")
        ax.set_ylabel("Fraction reaching this position")
        ax.legend(frameon=False)
    _save(fig, figures / "acceptance_survival.pdf")

    fig, ax = plt.subplots(figsize=(5.5, 2.2), layout="constrained")
    for i, target in enumerate(["8b", "14b"]):
        m = data["memory"][target]
        s = m["source_peak_allocated_bytes"] / 2**30
        r = m["relay_peak_allocated_bytes"] / 2**30
        ax.barh(i + 0.17, s, 0.30, color=BLUE, label="Source reuse" if i == 0 else None)
        ax.barh(
            i - 0.17,
            r,
            0.30,
            color=GREEN,
            label="RelaySpec" if i == 0 else None,
        )
        ax.text(s + 0.4, i + 0.17, f"{s:.2f}", va="center", fontsize=8)
        ax.text(r + 0.4, i - 0.17, f"{r:.2f}", va="center", fontsize=8)
    ax.set_yticks([0, 1], ["EAGLE-3 / 8B", "EAGLE-3 / 14B"])
    ax.invert_yaxis()
    ax.set_xlim(0, 43)
    ax.set_xlabel("Peak allocated GPU memory (GiB)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=2)
    _save(fig, figures / "memory_reduction.pdf")

    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x: float, y: float, w: float, label: str, color: str) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                0.75,
                boxstyle="round,pad=0.035",
                facecolor=color,
                edgecolor="0.35",
                linewidth=0.8,
            )
        )
        ax.text(x + w / 2, y + 0.375, label, ha="center", va="center", fontsize=9)

    def arrow(x: float, y: float, u: float, v: float) -> None:
        ax.annotate(
            "",
            xy=(u, v),
            xytext=(x, y),
            arrowprops={"arrowstyle": "->", "color": "0.3", "lw": 1},
        )

    ax.text(0, 3.75, "Source reuse", fontweight="bold", fontsize=10)
    box(0.05, 2.5, 2, "Source model\nfeatures", "#F1E3D6")
    box(3, 2.5, 2.5, "Frozen drafter\nproposes tokens", "#DDE9EF")
    box(6.5, 2.5, 3, "Target checks\nand commits tokens", "#DDEDE8")
    arrow(2.1, 2.875, 2.96, 2.875)
    arrow(5.55, 2.875, 6.46, 2.875)
    ax.text(0, 1.92, "RelaySpec", fontweight="bold", fontsize=10)
    box(0.05, 0.6, 2, "Target features\nfrom verification", "#DDEDE8")
    box(2.6, 0.6, 1.4, "Learned\nlinear map", "#C5DED6")
    box(4.55, 0.6, 2.1, "Same frozen\ndrafter", "#DDE9EF")
    box(7.2, 0.6, 2.3, "Target checks\nand commits", "#DDEDE8")
    arrow(2.1, 0.975, 2.56, 0.975)
    arrow(4.05, 0.975, 4.51, 0.975)
    arrow(6.7, 0.975, 7.16, 0.975)
    ax.plot([8.35, 8.35, 1.05, 1.05], [0.56, 0.15, 0.15, 0.56], color="0.4", lw=0.8)
    arrow(1.05, 0.30, 1.05, 0.56)
    ax.text(
        4.7,
        0.15,
        "Features at committed positions feed the next cycle",
        ha="center",
        va="center",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
    )
    _save(fig, figures / "system_overview.pdf")
    return data
