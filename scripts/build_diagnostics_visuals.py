"""Generate source-checked appendix diagnostics without new model evaluation.

Run: python scripts/build_diagnostics_visuals.py --root . --output paper/iclr2027
All curves retain their original workload, runtime, comparison denominator and
uncertainty convention. Registered raw files are hash checked and key quantities
are recomputed from requests. A sibling checkout may supply archived evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#78838D"
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": .65,
    "axes.edgecolor": "#57616A", "grid.color": "#DDE2E7",
    "grid.linewidth": .6, "pdf.fonttype": 42,
    "savefig.dpi": 200, "figure.facecolor": "white",
}
BASE = Path("reports/autoresearch-20260907")


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.inputs = {}

    def path(self, value):
        value = Path(value)
        relative = value
        if value.is_absolute():
            for component in ("reports", "configs", "src", "paper"):
                if f"/{component}/" in str(value):
                    relative = Path(component) / str(value).split(f"/{component}/", 1)[1]
                    break
        for path in (self.root / relative, value, self.root.parent / "RelaySpec" / relative):
            if path.is_file():
                return path.resolve()
        raise FileNotFoundError(value)

    def raw(self, value, expected=None):
        path = self.path(value)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f"Source hash mismatch: {path}")
        self.inputs[str(path)] = digest
        return raw

    def read(self, value, expected=None):
        return json.loads(self.raw(value, expected))

    def rows(self, value, expected=None):
        return [json.loads(line) for line in self.raw(value, expected).splitlines() if line.strip()]


def _save(fig, folder, name):
    fig.savefig(folder / f"{name}.pdf", metadata={
        "Creator": "RelaySpec build_diagnostics_visuals.py",
        "CreationDate": None, "ModDate": None,
    })
    fig.savefig(folder / f"{name}.png", metadata={"Software": "RelaySpec"})
    plt.close(fig)


def _key(row):
    return row["problem_id"], row.get("turn_index", 0), row.get("repetition", 0)


def _check_aggregate(rows, methods, n):
    for name, method in methods.items():
        selected = [row for row in rows if row["method"] == name]
        assert len(selected) == len({_key(row) for row in selected}) == n, name
        tps = sum(row["output_tokens"] for row in selected) / sum(row["request_seconds"] for row in selected)
        assert np.isclose(tps, method["tokens_per_second"], rtol=1e-10), name


def _complexity(evidence, folder):
    raw = evidence.read("reports/mapper-scaling-20260905/task-complexity-results.json")
    assert raw["status"] == "complete" and raw["requests"] == 128
    requests = []
    for path, digest in raw["input_sha256"].items():
        if "benchmark-rank" in path and path.endswith(".jsonl"):
            requests.extend(evidence.rows(path, digest))
        else:
            evidence.raw(path, digest)
    selected = [
        ("relay_dense_n512", "Dense, N=512", BLUE, "o"),
        ("relay_factorized1024_n512", "Linear 1024, N=512", GREEN, "s"),
        ("relay_factorized4096_n2048", "Linear 4096, N=2048", "#56B4E9", "D"),
        ("relay_mlp4096_n2048", "MLP 4096, N=2048", ORANGE, "^"),
        ("relay_factorized512_n2048", "Linear 512, N=2048", "#7B61A8", "v"),
        ("relay_mlp512_n2048", "MLP 512, N=2048", "#CC79A7", "P"),
    ]
    specs = [
        ("difficulty", ["Levels 1–2", "Level 3", "Levels 4–5"], "(a) Benchmark difficulty"),
        ("input_length", ["≤64", "65–128", "129–256", ">256"], "(b) Shared input length"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.65), sharey=True)
    fig.subplots_adjust(left=.10, right=.985, bottom=.37, top=.81, wspace=.17)
    values = {}
    for ax, (grouping, labels, title) in zip(axes, specs):
        groups = raw["strata"][grouping]
        values[grouping] = []
        ids = [item for group in groups.values() for item in group["problem_ids"]]
        assert len(ids) == len(set(ids)) == 128
        ticklabels = []
        for j, ((group_name, group), label) in enumerate(zip(groups.items(), labels, strict=True)):
            n = group["requests"]
            ticklabels.append(f"{label}\nn={n}")
            subset = [row for row in requests if row["problem_id"] in set(group["problem_ids"])]
            methods = group["comparisons"]["relay_dense_n2048"]["methods"]
            _check_aggregate(subset, methods, n)
            if n < 10:
                ax.axvspan(j-.48, j+.48, color="#FFF0D5", zorder=0)
            for k, (method, label, color, marker) in enumerate(selected):
                entry = methods[method]
                value = entry["throughput_ratio"] * 100
                lo, hi = np.asarray(entry["throughput_ci95"]) * 100
                ax.errorbar(j+(k-2.5)*.10, value, yerr=[[value-lo], [hi-value]],
                            color=color, marker=marker, linestyle="none", markersize=4.2,
                            capsize=1.4, linewidth=.95,
                            label=label if j == 0 and grouping == "difficulty" else None)
                values[grouping].append({"group": group_name, "requests": n,
                                         "method": method, **entry})
        ax.axhline(100, color=GRAY, linestyle="--", linewidth=.85)
        ax.set(xticks=range(len(groups)), xticklabels=ticklabels,
               xlim=(-.5, len(groups)-.5), ylim=(65, 104), yticks=[70, 80, 90, 100])
        ax.set_title(title, loc="left", fontweight="semibold", pad=9)
        ax.grid(axis="y", alpha=.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Throughput / dense N=2,048 (%)")
    axes[1].set_xlabel("Input tokens", labelpad=6)
    fig.text(.10, .92, "DFlash-8B · 128 MATH development questions · 2,048-token cap",
             fontsize=8.7, color="#45515B")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.51, .055),
               ncol=2, frameon=False, columnspacing=1.5, handletextpad=.4, labelspacing=.45)
    fig.text(.10, .017, "95% paired request intervals; shaded n=4 group is exploratory.",
             fontsize=8.5, color="#45515B")
    _save(fig, folder, "visual_diagnostics_complexity")
    return {"plotted": values, "all_strata": raw["strata"], "scope": raw["scope"],
            "selection": "Six original 8192-update dense/linear/MLP contrasts; complete controls, continuations and subject strata remain in the source and tables."}


def _quality(evidence, folder):
    raw = evidence.read("paper/iclr2027/generated/ar_evidence.json")
    points = []
    for entry in raw["main"]:
        run = entry["run"]
        request_rows = []
        scored_rows = []
        for path, digest in raw["source_sha256"].items():
            if f"/{run}/" in path:
                if "benchmark-rank" in path and path.endswith(".jsonl"):
                    request_rows.extend(evidence.rows(path, digest))
                elif path.endswith("math-scored.jsonl"):
                    scored_rows.extend(evidence.rows(path, digest))
                else:
                    evidence.raw(path, digest)
        assert entry["requests"] == 500
        _check_aggregate(request_rows, entry["methods"], 500)
        a = {_key(r): r for r in request_rows if r["method"] == "native_ar"}
        b = {_key(r): r for r in request_rows if r["method"] == entry["relay"]}
        matches = sum(a[k]["output_hash"] == b[k]["output_hash"] and
                      a[k]["output_tokens"] == b[k]["output_tokens"] for k in a)
        metric = entry["methods"][entry["relay"]]
        assert matches == metric["token_matches"]
        for method in ("native_ar", entry["relay"]):
            scores = [r for r in scored_rows if r["method"] == method]
            assert len(scores) == 500
            assert np.isclose(sum(r["correct"] for r in scores)/500,
                              entry["methods"][method]["accuracy"], rtol=1e-12)
            for scored in scores:
                timed = a[_key(scored)] if method == "native_ar" else b[_key(scored)]
                assert scored["output_hash"] == timed["output_hash"]
        assert np.isclose(metric["accuracy_difference"],
                          metric["accuracy"]-entry["methods"]["native_ar"]["accuracy"])
        points.append({"family": entry["family"], "target": entry["target"],
                       "requests": 500, "method": entry["relay"], "run": run, **metric})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.2, 2.8), sharey=True,
                                gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=.21, right=.98, bottom=.24, top=.73, wspace=.26)
    for i, point in enumerate(points):
        y = len(points)-i-1
        color = BLUE if point["family"] == "DFlash" else ORANGE
        value = point["accuracy_difference"] * 100
        lo, hi = np.asarray(point["accuracy_difference_ci95"]) * 100
        ax.errorbar(value, y, xerr=[[value-lo], [hi-value]], fmt="o",
                    color=color, markersize=5, capsize=2, linewidth=1.3)
        ax.text(value, y+.20, f"{value:+.1f}", color=color, ha="center", fontsize=8.5)
        pct = point["token_matches"] / point["requests"] * 100
        bx.barh(y, pct, color=color, height=.48)
        bx.text(pct+3, y, f"{point['token_matches']}/500", va="center", fontsize=8.5)
    ax.axvline(0, color=GRAY, linestyle="--", linewidth=.9)
    ax.set(yticks=range(4), yticklabels=[f"{p['family']} {p['target']}" for p in reversed(points)],
           xlim=(-3.1, 3.8), xticks=[-2, 0, 2], ylim=(-.55, 3.6),
           xlabel="Relay − AR accuracy (pp)")
    bx.set(xlim=(0, 100), xticks=[0, 50, 100], xlabel="Exact AR sequences (%)")
    ax.set_title("(a) Task accuracy", loc="left", fontweight="semibold", pad=10)
    bx.set_title("(b) Token agreement", loc="left", fontweight="semibold", pad=10)
    for axis in (ax, bx):
        axis.grid(axis="x", alpha=.8)
        axis.set_axisbelow(True)
    fig.text(.21, .92, "Primary BF16 Qwen suite · MATH-500 · 2,048-token cap",
             fontsize=8.5, color="#45515B")
    fig.text(.21, .033, "Paired 95% accuracy intervals; token agreement is a different metric.",
             fontsize=8.3, color="#45515B")
    _save(fig, folder, "visual_diagnostics_quality")
    return {"plotted": points, "scope": "Primary Qwen BF16/SDPA runs only; paired request-bootstrap accuracy intervals, hash/length token equality; not pooled with exact FP32-target family or batch-invariant BF16 rollout extensions."}


def _memory(evidence, folder):
    raw = evidence.read(BASE / "capture-memory-summary.json")
    points = []
    for point in raw["results"]:
        path = BASE / f"run-{point['job']}" / f"lane{point['lane']}" / "benchmark-rank0.jsonl"
        rows = evidence.rows(path, raw["input_sha256"][str(path)])
        groups = {method: {_key(row): row for row in rows if row["method"] == method}
                  for method in ("relay_base", "relay_selected")}
        a, b = groups.values()
        assert a.keys() == b.keys() and len(a) == 2
        for key in a:
            for field in ("output_hash", "output_tokens", "acceptance_lengths", "target_calls", "draft_calls", "input_tokens", "mapper_checkpoint_sha256"):
                assert a[key][field] == b[key][field]
        for method, group in groups.items():
            peaks = [r["peak_allocated_memory_bytes"] / 2**30 for r in group.values()]
            assert np.allclose(peaks, point["peak_allocated_gib"][method], rtol=1e-12)
        saved = [(a[key]["peak_allocated_memory_bytes"]-b[key]["peak_allocated_memory_bytes"])/2**30 for key in a]
        assert np.allclose(saved, point["saved_gib"], rtol=1e-12)
        points.append({**point, "all_states_gib": float(np.mean(point["peak_allocated_gib"]["relay_base"])),
                       "selected_states_gib": float(np.mean(point["peak_allocated_gib"]["relay_selected"])),
                       "mean_saved_gib": float(np.mean(point["saved_gib"]))})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.2, 3.2))
    fig.subplots_adjust(left=.10, right=.985, bottom=.29, top=.78, wspace=.33)
    for taps, color, marker in [(5, BLUE, "o"), (2, ORANGE, "s")]:
        data = sorted((p for p in points if p["taps"] == taps and not p["prefill_lifetime_control"]), key=lambda p: p["input_tokens"])
        assert len(data) == 4
        x = [p["input_tokens"] / 1000 for p in data]
        for field, style, label in [("all_states_gib", "--", "All states"), ("selected_states_gib", "-", "Selected")]:
            ax.plot(x, [p[field] for p in data], color=color, marker=marker,
                    linestyle=style, linewidth=1.3, markersize=4.2, label=f"{taps} taps: {label}")
        bx.plot(x, [p["mean_saved_gib"] for p in data], color=color, marker=marker,
                linewidth=1.4, markersize=4.2, label=f"{taps} taps")
        controls = [p for p in points if p["taps"] == taps and p["prefill_lifetime_control"]]
        bx.scatter([p["input_tokens"] / 1000 for p in controls], [p["mean_saved_gib"] for p in controls],
                   s=68, facecolors="none", edgecolors=color, linewidths=1.1,
                   label="Release-prefill control" if taps == 5 else None, zorder=4)
    for axis in (ax, bx):
        axis.set(xlabel="Input tokens (thousands)", xlim=(0, 34), xticks=[0, 8, 16, 24, 32])
        axis.grid(alpha=.8)
        axis.set_axisbelow(True)
    ax.set(ylabel="Peak allocated memory (GiB)", ylim=(16, 35))
    bx.set(ylabel="Allocated memory saved (GiB)", ylim=(0, 8.6))
    ax.set_title("(a) Retain only selected features", loc="left", fontweight="semibold", pad=10)
    bx.set_title("(b) Savings persist after release", loc="left", fontweight="semibold", pad=10)
    fig.text(.10, .93, "DFlash-8B · L40S · one synthetic prompt per length · 64-token cap",
             fontsize=8.5, color="#45515B")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.085, .04), ncol=2,
               frameon=False, handlelength=1.7, columnspacing=.9, labelspacing=.35)
    fig.text(.62, .075, "Open rings: release-prefill control\nTwo reversed-order repetitions;\nnot independent prompt samples.", fontsize=8.1, color="#45515B")
    _save(fig, folder, "visual_diagnostics_memory")
    return {"plotted": points, "scope": raw["scope"]}


def _native(evidence, folder):
    specifications = [
        (28524, "DFlash", [("native_target_dflash", "Released", 5*4096*4096),
                           ("relay_base", "Repacked", 5*4096*4096),
                           ("relay_two", "2-layer fit", 2*4096*4096),
                           ("relay_one", "1-layer fit", 4096*4096),
                           ("relay_svd1536", "Rank 1,536", 1536*(5*4096+4096))]),
        (28560, "EAGLE-3", [("native_target_eagle3", "Released", 5*4096*4096),
                             ("relay_base", "Repacked", 5*4096*4096),
                             ("relay_two", "2-layer long fit", 2*4096*4096),
                             ("relay_short", "2-layer short fit", 2*4096*4096),
                             ("relay_svd1536", "Rank 1,536", 1536*(5*4096+4096))]),
    ]
    points = []
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.6))
    fig.subplots_adjust(left=.235, right=.985, bottom=.28, top=.78, wspace=1.20)
    for ax, (job, family, methods) in zip(axes, specifications):
        raw = evidence.read(BASE / f"run-{job}" / "confirmation-analysis.json")
        assert raw["status"] == "complete" and raw["summary"]["requests"] == 64
        rows = []
        for path, digest in raw["input_sha256"].items():
            if "benchmark-rank" in path and path.endswith(".jsonl"):
                rows.extend(evidence.rows(path, digest))
            else:
                evidence.raw(path, digest)
        _check_aggregate(rows, raw["summary"]["methods"], 64)
        color = BLUE if family == "DFlash" else GREEN
        for i, (name, label, parameters) in enumerate(methods):
            row = raw["summary"]["methods"][name]
            value = 100 * row["throughput_ratio"]
            lo, hi = np.asarray(row["throughput_ci95"]) * 100
            ax.errorbar(value, 4-i, xerr=[[value-lo], [hi-value]], fmt="o" if "layer" in label else "s",
                        color=color, markersize=4.5, capsize=1.8, linewidth=1.2)
            points.append({"job": job, "family": family, "method": name,
                           "label": label, "parameters": parameters, "requests": 64, **row})
        ax.axvline(100, color=GRAY, linestyle="--", linewidth=.9)
        ax.axvline(95, color="#BEC5CB", linestyle=":", linewidth=.9)
        ax.set(xlim=(80, 103), xticks=[80, 90, 100], ylim=(-.5, 4.6),
               yticks=range(5), yticklabels=[f"{label}\n{p/1e6:.1f}M" for _, label, p in reversed(methods)],
               xlabel="Native retention (%)")
        ax.grid(axis="x", alpha=.8)
        ax.set_axisbelow(True)
        ax.set_title(family, loc="left", fontweight="semibold", pad=11)
    fig.text(.07, .94, "Native Qwen3-8B projection · separate frozen 64-question GSM8K sets", fontsize=8.5, color="#45515B")
    fig.text(.07, .048, "95% paired request intervals · 2,048-token cap · counts cover only the projection.\nDotted line: 95% retention criterion. Layers and optimization differ across controls.", fontsize=8.3, color="#45515B")
    _save(fig, folder, "visual_diagnostics_native")
    return {"plotted": points, "parameter_formula": "Dense k-tap projection: k*4096*4096; rank-r factorization: r*(5*4096+4096)",
            "scope": "Separate frozen 64-question GSM8K sets, 2048 output cap; own-native end-to-end throughput. DFlash one/two-layer students and EAGLE inherited initialization with short/long fitting; all five arms shown in each experiment. Parameter counts are projection-only, not full deployment memory."}


def build(root, output):
    evidence = Evidence(root)
    output = Path(output)
    folder = output / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    (output / "generated").mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        result = {"complexity": _complexity(evidence, folder),
                  "quality": _quality(evidence, folder),
                  "memory": _memory(evidence, folder),
                  "native_compression": _native(evidence, folder)}
    registry = {"status": "complete", "input_sha256": dict(sorted(evidence.inputs.items())),
                "plotted_values": result, "figures": [p.name for p in sorted(folder.glob("visual_diagnostics*.pdf"))],
                "scope": "Retrospective visualizations of completed experiments; no additional fitting, generation, selection or pooling across numerical runtimes."}
    (output / "generated/diagnostics_visuals_evidence.json").write_text(json.dumps(registry, indent=2, sort_keys=True)+"\n")
    return registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("paper/iclr2027"))
    args = parser.parse_args()
    result = build(args.root, args.output)
    print(json.dumps({"status": result["status"], "figures": result["figures"], "checked_files": len(result["input_sha256"])}, indent=2))


if __name__ == "__main__":
    main()
