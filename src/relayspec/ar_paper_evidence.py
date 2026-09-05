"""Reconstruct AR-primary paper comparisons from paired request records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

RUNS = [
    (
        "DFlash",
        "8B",
        "bench-dflash-8b-pairedAR-v2-27357",
        "optimized_source_reuse",
        "relay_p",
        "native_target_dflash",
    ),
    (
        "DFlash",
        "14B",
        "bench-dflash-14b-pairedAR-27325",
        "optimized_source_reuse",
        "relay_p",
        None,
    ),
    (
        "EAGLE-3",
        "8B",
        "bench-eagle3-8b-pairedAR-v3-27396",
        "source_reuse_eagle3",
        "relay_eagle3",
        "native_target_eagle3",
    ),
    (
        "EAGLE-3",
        "14B",
        "bench-eagle3-14b-pairedAR-v3-27397",
        "source_reuse_eagle3",
        "relay_eagle3",
        "native_target_eagle3",
    ),
]


def read_rows(directory: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for p in sorted(directory.glob("benchmark-rank*.jsonl"))
        for line in p.open()
        if line.strip()
    ]


def summarize(
    rows: list[dict[str, Any]], reference: str = "native_ar", samples: int = 10000
) -> dict[str, Any]:
    groups = {}
    methods = sorted({r["method"] for r in rows})
    if reference not in methods:
        raise ValueError("The stated reference was not measured")
    for r in rows:
        key = (r["problem_id"], int(r.get("repetition", 0)))
        g = groups.setdefault(key, {})
        if r["method"] in g:
            raise ValueError("Duplicate method/request record")
        if (
            not np.isfinite(r["request_seconds"])
            or r["request_seconds"] <= 0
            or not np.isfinite(r["output_tokens"])
            or r["output_tokens"] <= 0
        ):
            raise ValueError("Invalid recorded time or token count")
        g[r["method"]] = r
    if not groups or any(set(g) != set(methods) for g in groups.values()):
        raise ValueError("Incomplete paired methods")
    clusters = {}
    for key, group in sorted(groups.items()):
        first = group[reference]
        pid = (
            key[0].rsplit("/turn", 1)[0] if first["benchmark"] == "mtbench" else key[0]
        )
        cluster = clusters.setdefault((pid, key[1]), {m: [0.0, 0.0] for m in methods})
        for m in methods:
            cluster[m][0] += group[m]["output_tokens"]
            cluster[m][1] += group[m]["request_seconds"]
    arrays = {m: np.array([g[m] for g in clusters.values()]) for m in methods}
    indices = np.random.default_rng(1729).integers(
        0, len(clusters), (samples, len(clusters))
    )
    sampled = {m: a[indices].sum(axis=1) for m, a in arrays.items()}
    ref = arrays[reference].sum(axis=0)
    answer = {
        "requests": len(groups),
        "clusters": len(clusters),
        "reference": reference,
        "methods": {},
    }
    for m, a in arrays.items():
        tokens, seconds = a.sum(axis=0)
        ratios = (sampled[m][:, 0] / sampled[m][:, 1]) / (
            sampled[reference][:, 0] / sampled[reference][:, 1]
        )
        time_ratios = sampled[reference][:, 1] / sampled[m][:, 1]
        mr = [g[m] for g in groups.values()]
        lengths = [n for r in mr for n in r.get("acceptance_lengths", [])]
        q = {
            "tokens_per_second": float(tokens / seconds),
            "throughput_ratio": float((tokens / seconds) / (ref[0] / ref[1])),
            "throughput_ci95": np.quantile(ratios, [0.025, 0.975]).tolist(),
            "request_time_ratio": float(ref[1] / seconds),
            "request_time_ci95": np.quantile(time_ratios, [0.025, 0.975]).tolist(),
            "mean_output_tokens": float(tokens / len(groups)),
            "token_matches": sum(
                g[m]["output_hash"] == g[reference]["output_hash"]
                for g in groups.values()
            ),
            "progress_per_cycle": float(np.mean(lengths))
            if lengths
            else float(np.mean([r["acceptance_length"] for r in mr])),
            "request_seconds": float(seconds),
        }
        if all(r.get("correct") is not None for r in mr):
            diffs = np.array(
                [
                    int(g[m]["correct"]) - int(g[reference]["correct"])
                    for g in groups.values()
                ]
            )
            if len(clusters) != len(groups):
                raise ValueError(
                    "Scored conversation bootstrap needs explicit task outcomes"
                )
            q.update(
                {
                    "accuracy": float(np.mean([r["correct"] for r in mr])),
                    "accuracy_difference": float(diffs.mean()),
                    "accuracy_difference_ci95": np.quantile(
                        diffs[indices].mean(1), [0.025, 0.975]
                    ).tolist(),
                }
            )
        answer["methods"][m] = q
    return answer


def load_scored(directory: Path) -> list[dict[str, Any]]:
    raw = read_rows(directory)
    scored = [
        json.loads(line)
        for line in (directory / "math-scored.jsonl").open()
        if line.strip()
    ]
    lookup = {(r["problem_id"], r["repetition"], r["method"]): r for r in raw}
    seen = set()
    for r in scored:
        key = (r["problem_id"], r["repetition"], r["method"])
        if key in seen or key not in lookup:
            raise ValueError("Duplicate or unmatched score")
        seen.add(key)
        if any(r[k] != v for k, v in lookup[key].items()):
            raise ValueError("Scored row differs from raw generation")
    if len(scored) != sum(r["benchmark"] in {"math500", "gsm8k"} for r in raw):
        raise ValueError("Missing task scores")
    return scored


def build_ar_assets(root: Path, output: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    from relayspec.paper_evidence import BLUE, GREEN, ORANGE, _save, _table

    base = root / "reports/ar-revision-20260905/raw"
    generated, figures = output / "generated", output / "figures"
    main = []
    for family, target, run, source, relay, native in RUNS:
        rows = load_scored(base / run)
        result = summarize(rows)
        if result["requests"] != 500:
            raise ValueError("Main comparison must contain 500 paired requests")
        result.update(
            family=family,
            target=target,
            run=run,
            source=source,
            relay=relay,
            native=native,
        )
        result["secondary"] = summarize(rows, source)
        if native is not None:
            result["native_comparison"] = summarize(rows, native)
        main.append(result)
    native_rows = []
    for x in main:
        if x["native"] is None:
            continue
        q = x["native_comparison"]["methods"][x["relay"]]
        native_rows.append(
            f"{x['family']} & {x['target']} & {x['methods'][x['native']]['tokens_per_second']:.2f} & {q['tokens_per_second']:.2f} & {100 * q['throughput_ratio']:.1f}\\% "
            f"[{100 * q['throughput_ci95'][0]:.1f}, {100 * q['throughput_ci95'][1]:.1f}] & "
            f"{x['methods'][x['native']]['progress_per_cycle']:.2f} & {q['progress_per_cycle']:.2f}"
        )
    _table(
        generated / "native_retention_table.tex",
        "llrrrrr",
        r"Drafter & Target & Native tok/s & Relay tok/s & Retained [95\% CI] & Progress N & Progress R",
        native_rows,
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), layout="constrained")
    native_main = [x for x in main if x["native"] is not None]
    for i, x in enumerate(native_main):
        q = x["native_comparison"]["methods"][x["relay"]]
        v = 100 * q["throughput_ratio"]
        lo, hi = np.array(q["throughput_ci95"]) * 100
        axes[0].barh(i, v, color=GREEN)
        axes[0].errorbar(
            v, i, xerr=[[v - lo], [hi - v]], fmt="none", ecolor="black", capsize=3
        )
        axes[0].text(
            v - 2, i, f"{v:.1f}%", ha="right", va="center", color="white", fontsize=9
        )
    axes[0].set_yticks(
        range(3), [f"{x['family']} / {x['target']}" for x in native_main]
    )
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Native-drafter throughput retained (%)")
    axes[1].barh([0, 1, 2], [800000, 532000, 4096], color=[ORANGE, BLUE, GREEN])
    axes[1].set_yticks([0, 1, 2], ["DFlash recipe", "EAGLE-3 recipe", "Relay fitting"])
    axes[1].invert_yaxis()
    axes[1].set_xscale("log")
    axes[1].set_xlim(1000, 3000000)
    for i, v in enumerate([800000, 532000, 4096]):
        axes[1].text(v * 1.15, i, f"{v:,}", va="center", fontsize=8)
    axes[1].set_xlabel("Training records (log scale)")
    _save(fig, figures / "reuse_tradeoff.pdf")
    _table(
        generated / "ar_main_table.tex",
        "llrrrrr",
        r"Drafter & Target & AR tok/s & Source tok/s & Relay tok/s & Relay / AR [95\% CI] & Progress R",
        [
            f"{x['family']} & {x['target']} & {x['methods']['native_ar']['tokens_per_second']:.2f} & {x['methods'][x['source']]['tokens_per_second']:.2f} & "
            + f"{x['methods'][x['relay']]['tokens_per_second']:.2f} & {x['methods'][x['relay']]['throughput_ratio']:.2f} [{x['methods'][x['relay']]['throughput_ci95'][0]:.2f}, {x['methods'][x['relay']]['throughput_ci95'][1]:.2f}] & {x['methods'][x['relay']]['progress_per_cycle']:.2f}"
            for x in main
        ],
    )
    _table(
        generated / "ar_quality_table.tex",
        "llrrrr",
        r"Drafter & Target & AR score & Relay score & Difference [95\% CI] & Token matches",
        [
            f"{x['family']} & {x['target']} & {100 * x['methods']['native_ar']['accuracy']:.1f} & {100 * x['methods'][x['relay']]['accuracy']:.1f} & "
            + f"{100 * x['methods'][x['relay']]['accuracy_difference']:+.1f} [{100 * x['methods'][x['relay']]['accuracy_difference_ci95'][0]:+.1f}, {100 * x['methods'][x['relay']]['accuracy_difference_ci95'][1]:+.1f}] & {x['methods'][x['relay']]['token_matches']}/500"
            for x in main
        ],
    )
    _table(
        generated / "ar_details_table.tex",
        "llrrrr",
        "Drafter & Target & Mean tokens AR / R & Time ratio AR / R & Throughput R / S & Progress S / R",
        [
            f"{x['family']} & {x['target']} & {x['methods']['native_ar']['mean_output_tokens']:.2f} / {x['methods'][x['relay']]['mean_output_tokens']:.2f} & {x['methods'][x['relay']]['request_time_ratio']:.3f} & {x['secondary']['methods'][x['relay']]['throughput_ratio']:.3f} & {x['methods'][x['source']]['progress_per_cycle']:.2f} / {x['methods'][x['relay']]['progress_per_cycle']:.2f}"
            for x in main
        ],
    )
    fig, ax = plt.subplots(figsize=(6.6, 2.65), layout="constrained")
    y = np.arange(4)
    for offset, key, label, color in [
        (-0.27, "ar", "AR", "#BBBBBB"),
        (-0.09, "source", "Source reuse", BLUE),
        (0.09, "native", "Target-specific drafter", ORANGE),
        (0.27, "relay", "RelaySpec", GREEN),
    ]:
        for i, x in enumerate(main):
            m = "native_ar" if key == "ar" else x[key]
            if m is None:
                continue
            q = x["methods"][m]
            ax.barh(
                i + offset,
                q["throughput_ratio"],
                0.17,
                color=color,
                label=label if i == 0 else None,
            )
            if key == "relay":
                lo, hi = q["throughput_ci95"]
                v = q["throughput_ratio"]
                ax.errorbar(
                    v,
                    i + offset,
                    xerr=[[v - lo], [hi - v]],
                    fmt="none",
                    ecolor="black",
                    capsize=2,
                )
                label_start = max(
                    hi,
                    *(m["throughput_ratio"] for m in x["methods"].values()),
                )
                ax.annotate(
                    f"{v:.2f}x",
                    xy=(label_start, i + offset),
                    xytext=(8, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8,
                )
    ax.set_yticks(y, [f"{x['family']} / {x['target']}" for x in main])
    ax.invert_yaxis()
    ax.axvline(1, color=".5", lw=0.8, ls="--")
    ax.set_xlim(0, 6.3)
    ax.set_xlabel("End-to-end throughput relative to AR (AR = 1)")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.27), frameon=False)
    _save(fig, figures / "ar_main_throughput.pdf")
    breadth = []
    for target, run in [
        ("8B", "breadth-ar-dflash8b-27498"),
        ("14B", "breadth-ar-dflash14b-27499"),
    ]:
        rows = read_rows(base / run)
        for task in ["gsm8k", "humaneval", "mbpp", "mtbench"]:
            q = summarize([r for r in rows if r["benchmark"] == task])
            q.update(target=target, task=task, run=run)
            breadth.append(q)
    labels = {
        "gsm8k": "GSM8K",
        "humaneval": "HumanEval",
        "mbpp": "MBPP",
        "mtbench": "MT-Bench",
    }
    _table(
        generated / "ar_breadth_table.tex",
        "llrrrrr",
        r"Target & Task & Requests & AR tok/s & Relay tok/s & Relay / AR [95\% CI] & Relay / source",
        [
            f"{x['target']} & {labels[x['task']]} & {x['requests']} & {x['methods']['native_ar']['tokens_per_second']:.2f} & {x['methods']['relay_p']['tokens_per_second']:.2f} & {x['methods']['relay_p']['throughput_ratio']:.2f} [{x['methods']['relay_p']['throughput_ci95'][0]:.2f}, {x['methods']['relay_p']['throughput_ci95'][1]:.2f}] & {x['methods']['relay_p']['tokens_per_second'] / x['methods']['optimized_source_reuse']['tokens_per_second']:.3f}"
            for x in breadth
        ],
    )
    fig, ax = plt.subplots(figsize=(6.6, 2.5), layout="constrained")
    for j, target in enumerate(["8B", "14B"]):
        rs = [x for x in breadth if x["target"] == target]
        for i, x in enumerate(rs):
            q = x["methods"]["relay_p"]
            v = q["throughput_ratio"]
            lo, hi = q["throughput_ci95"]
            ax.errorbar(
                i + (j - 0.5) * 0.18,
                v,
                yerr=[[v - lo], [hi - v]],
                fmt=["o", "s"][j],
                color=[GREEN, BLUE][j],
                capsize=3,
                label=target if i == 0 else None,
            )
    ax.axhline(1, color=".5", ls="--")
    ax.set_xticks(range(4), list(labels.values()))
    ax.set_ylabel("RelaySpec throughput / AR")
    ax.legend(title="DFlash target", frameon=False, ncol=2)
    ax.set_ylim(bottom=0)
    _save(fig, figures / "ar_breadth.pdf")
    blocks = []
    for b in [8, 16, 32]:
        q = summarize(read_rows(root / f"reports/d8-block{b}"))
        q["block"] = b
        blocks.append(q)
    _table(
        generated / "block_ablation_table.tex",
        "rrrrrr",
        r"Block & AR tok/s & Native draft tok/s & Relay tok/s & Relay / AR & Progress / cycle",
        [
            f"{x['block']} & {x['methods']['native_ar']['tokens_per_second']:.2f} & {x['methods']['native_target_dflash']['tokens_per_second']:.2f} & {x['methods']['relay_f']['tokens_per_second']:.2f} & {x['methods']['relay_f']['throughput_ratio']:.3f} & {x['methods']['relay_f']['progress_per_cycle']:.2f}"
            for x in blocks
        ],
    )
    fit_specs = [
        ("512 records", "bench-scale512c-27116", 512, 128),
        ("1,024 records", "bench-scale1024c-27118", 1024, 256),
        ("2,048 records", "bench-scale2048c-27120", 2048, 512),
        ("4,096 records, two passes", "bench-scale8192c-27122", 4096, 2048),
        ("Ridge surrogate", "bench-closed-form-fixed-27422", 4096, None),
        ("Nonlinear width 512", "bench-mlp512c-27115", 4096, 1024),
    ]
    fits = []
    for label, run, n, steps in fit_specs:
        q = summarize(load_scored(base / run), "optimized_source_reuse")
        q.update(label=label, run=run, distinct_records=n, updates=steps)
        fits.append(q)
    _table(
        generated / "fit_ablation_table.tex",
        "lrrrrr",
        r"Fitting choice & Records & Updates & Relay tok/s & Relay / source [95\% CI] & Score",
        [
            f"{x['label']} & {x['distinct_records']:,} & "
            + (f"{x['updates']:,}" if x["updates"] else r"\textit{solve}")
            + f" & {x['methods']['relay_p']['tokens_per_second']:.2f} & {x['methods']['relay_p']['throughput_ratio']:.3f} [{x['methods']['relay_p']['throughput_ci95'][0]:.3f}, {x['methods']['relay_p']['throughput_ci95'][1]:.3f}] & {100 * x['methods']['relay_p']['accuracy']:.2f}"
            for x in fits
        ],
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), layout="constrained")
    scale = fits[:4]
    presentations = [x["updates"] * 4 for x in scale]
    axes[0].plot(
        presentations,
        [x["methods"]["relay_p"]["tokens_per_second"] for x in scale],
        "o-",
        color=GREEN,
    )
    axes[0].set_ylabel("RelaySpec output tokens / second")
    axes[1].plot(
        presentations,
        [x["methods"]["relay_p"]["progress_per_cycle"] for x in scale],
        "s-",
        color=BLUE,
    )
    axes[1].set_ylabel("Recorded tokens advanced / cycle")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(presentations, ["512", "1,024", "2,048", "8,192"])
        ax.set_xlabel("Training record presentations")
        ax.grid(alpha=0.15)
    _save(fig, figures / "fitting_scale.pdf")
    report = {
        "main": main,
        "breadth": breadth,
        "blocks": blocks,
        "fitting": fits,
        "bootstrap": {
            "replicates": 10000,
            "seed": 1729,
            "unit": "request, or whole MT-Bench conversation",
        },
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(base.rglob("*"))
            if p.is_file()
        },
    }
    (generated / "ar_evidence.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    draw_method_diagrams(figures)
    return report


def draw_method_diagrams(figures: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    from relayspec.paper_evidence import _save

    def box(ax, x, y, width, label, color):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                0.68,
                boxstyle="round,pad=0.025",
                facecolor=color,
                edgecolor=".4",
                linewidth=0.8,
            )
        )
        ax.text(x + width / 2, y + 0.34, label, ha="center", va="center", fontsize=8.5)

    def arrow(ax, x, y, u, v):
        ax.annotate(
            "",
            xy=(u, v),
            xytext=(x, y),
            arrowprops={"arrowstyle": "->", "color": ".3", "lw": 1},
        )

    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.25)
    ax.axis("off")
    ax.text(0, 4.05, "Fit once for the new target", fontweight="bold", fontsize=10)
    box(ax, 0.1, 2.65, 2, "Shared text", "#ECECEC")
    box(ax, 3, 3.15, 2.4, "Frozen source\ninput projection", "#F1E3D6")
    box(ax, 3, 2.05, 2.4, "Frozen target\nhidden states", "#DDEDE8")
    box(ax, 6.25, 2.05, 1.7, "Trainable\nlinear map", "#C5DED6")
    box(ax, 9, 2.65, 2.6, "Match drafter\ninput context", "#DDE9EF")
    arrow(ax, 2.15, 3.0, 2.95, 3.48)
    arrow(ax, 2.15, 3.0, 2.95, 2.38)
    arrow(ax, 5.45, 2.39, 6.2, 2.39)
    arrow(ax, 8, 2.39, 8.95, 2.94)
    arrow(ax, 5.45, 3.49, 8.95, 3.05)
    ax.text(0, 1.55, "Generate with the new target", fontweight="bold", fontsize=10)
    for x, w, label, col in [
        (0.1, 2.2, "Target features", "#DDEDE8"),
        (3, 1.8, "Fitted map", "#C5DED6"),
        (5.6, 2.3, "Frozen drafter", "#DDE9EF"),
        (8.8, 2.8, "Target verification", "#DDEDE8"),
    ]:
        box(ax, x, 0.5, w, label, col)
    for a, b in [(2.35, 2.95), (4.85, 5.55), (7.95, 8.75)]:
        arrow(ax, a, 0.84, b, 0.84)
    ax.plot([10.2, 10.2, 1.2, 1.2], [0.46, 0.12, 0.12, 0.46], color=".4", lw=0.8)
    arrow(ax, 1.2, 0.25, 1.2, 0.46)
    _save(fig, figures / "retargeting_workflow.pdf")
    fig, ax = plt.subplots(figsize=(6.6, 1.55))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2)
    ax.axis("off")
    specs = [
        (0.05, 2.25, "Propose draft\ntokens", "#DDE9EF"),
        (3.1, 2.2, "Target checks\nall positions", "#DDEDE8"),
        (6.1, 2.7, "Accept draft prefix\nand target token", "#DDEDE8"),
        (9.5, 2.35, "Crop cache\nand map states", "#C5DED6"),
    ]
    for x, w, label, col in specs:
        box(ax, x, 0.85, w, label, col)
    for a, b in [(2.35, 3.05), (5.35, 6.05), (8.85, 9.45)]:
        arrow(ax, a, 1.19, b, 1.19)
    ax.plot([10.65, 10.65, 1.2, 1.2], [0.81, 0.25, 0.25, 0.81], color=".4", lw=0.8)
    arrow(ax, 1.2, 0.55, 1.2, 0.81)
    ax.text(
        6,
        0.25,
        "Repeat until an end token or the output limit",
        ha="center",
        va="center",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none"},
    )
    _save(fig, figures / "verification_flow.pdf")
