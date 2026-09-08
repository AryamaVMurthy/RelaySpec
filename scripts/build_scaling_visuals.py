"""Build compact, source-checked data/capacity figures from completed evaluations.

Usage: python scripts/build_scaling_visuals.py --root . --output paper/iclr2027
The sibling RelaySpec checkout is a read-only fallback for archived raw evidence,
matching the existing manuscript builders. Every consumed file is hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import yaml


COLORS = {"dense": "#0072B2", "factorized": "#009E73", "mlp": "#D55E00"}
MARKERS = {"dense": "o", "factorized": "s", "mlp": "^"}
LABELS = {"dense": "Dense linear", "factorized": "Factored linear", "mlp": "GELU MLP"}
BASE = Path("reports/mapper-scaling-20260905")


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.inputs = {}

    def path(self, value):
        p = Path(value)
        if p.is_absolute():
            text = str(p)
            if "/reports/" in text:
                relative = Path("reports") / text.split("/reports/", 1)[1]
            elif "/configs/" in text:
                relative = Path("configs") / text.split("/configs/", 1)[1]
            else:
                relative = p
        else:
            relative = p
        for candidate in [self.root / relative, p, self.root.parent / "RelaySpec" / relative]:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(value)

    def read(self, value, expected=None, jsonl=False):
        path = self.path(value)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f"Source hash mismatch: {path}")
        self.inputs[str(path)] = digest
        if jsonl:
            return [json.loads(line) for line in raw.splitlines() if line.strip()]
        if path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(raw)
        return json.loads(raw)


def checked_analysis(evidence, folder, expected=None):
    config = evidence.read(folder / "config.yaml")
    assert config["generation"]["max_new_tokens"] == 2048
    assert config["generation"]["temperature"] == 0
    assert config["benchmark"]["max_prompts"] == 128
    assert config["target"]["id"] == "Qwen/Qwen3-8B"
    assert config["proposer"]["family"] == "dflash"
    analysis = evidence.read(folder / "analysis.json", expected)
    if analysis["status"] != "complete":
        raise ValueError(f"Incomplete analysis: {folder}")
    rows = []
    for name, digest in analysis["raw_sha256"].items():
        rows.extend(evidence.read(folder / name, digest, jsonl=True))
    task = analysis["tasks"]["math500"]
    assert task["requests"] == 128
    ar_ids = {
        (r["problem_id"], r["turn_index"], r["repetition"])
        for r in rows if r["method"] == "native_ar"
    }
    assert len(ar_ids) == 128
    for method, metric in task["methods"].items():
        selected = [r for r in rows if r["method"] == method]
        ids = {(r["problem_id"], r["turn_index"], r["repetition"]) for r in selected}
        assert len(selected) == 128 and ids == ar_ids
        tps = sum(r["output_tokens"] for r in selected) / sum(r["request_seconds"] for r in selected)
        if not np.isclose(tps, metric["tokens_per_second"], rtol=1e-10):
            raise ValueError(f"Reported throughput differs from raw requests: {method}")
    return task["methods"]


def load_points(evidence):
    data_summary = evidence.read("reports/data-small-20260906/results.json")
    assert data_summary["status"] == "complete"
    for path, digest in data_summary["source_sha256"].items():
        evidence.read(path, digest)
    methods = checked_analysis(evidence, Path("reports/data-small-20260906/run-28339"))
    data = []
    for n, metric in data_summary["panels"]["fixed_8192_updates"].items():
        assert metric == methods[f"relay_n{n}_fixed"]
        data.append({"records": int(n), **metric})
    data.sort(key=lambda p: p["records"])
    assert [p["records"] for p in data] == [16, 32, 64, 128, 256, 512, 2048, 4096, 8192, 16384, 32768]

    quality = evidence.read(BASE / "small-data-quality-results.json")
    assert quality["status"] == "complete" and quality["output_cap"] == 2048
    folder = Path(quality["run"])
    cap_methods = checked_analysis(evidence, folder, quality["analysis_sha256"])
    for method, summary in quality["against_ar"]["methods"].items():
        assert all(cap_methods[method][key] == value for key, value in summary.items())
    fits = evidence.read(BASE / "primary-capacity-results.json")["results"]
    for point in data:
        n = point["records"]
        if n <= 256:
            fit_path = Path("reports/data-small-20260906/run-28337/fitting") / f"dense-n{n}-s1729/fit-complete.json"
        elif n in {512, 2048}:
            fit_path = Path(fits[f"dense-n{n}-s1729"]["raw_directory"]) / "fit-complete.json"
        else:
            fit_path = Path("reports/data-scaling-20260906/run-28272/fitting") / f"dense-n{n}-s1729/fit-complete.json"
        fit = evidence.read(fit_path)
        assert fit["trial"]["steps"] == 8192
        assert fit["trial"]["distinct_examples"] == n
        assert fit["distinct_records_seen"] == n
    continuations = evidence.read(BASE / "continued-small-data-results.json")
    assert continuations["status"] == "complete"
    specs = [
        ("dense", None, 512), ("dense", None, 2048),
        ("factorized", 1024, 512), ("factorized", 4096, 2048),
        ("factorized", 512, 2048), ("mlp", 4096, 2048), ("mlp", 512, 2048),
    ]
    capacity = []
    for family, width, n in specs:
        key = family + (str(width) if width is not None else "")
        fit = fits[f"{key}-n{n}-s1729"]
        raw_fit = evidence.read(Path(fit["raw_directory"]) / "fit-complete.json", fit["fit_sha256"])
        trajectory = evidence.read(Path(fit["raw_directory"]) / "validation.jsonl", fit["validation_sha256"], jsonl=True)
        assert raw_fit["trial"]["steps"] == 8192
        last = next(v for v in trajectory if v["step"] == 8192)
        assert last["groups"]["train"]["objective"] == fit["train_objective"]
        assert last["groups"]["validation"]["objective"] == fit["validation_objective"]
        method = f"relay_{key}_n{n}"
        capacity.append({
            "method": method, "family": family, "width": width, "records": n,
            "updates": 8192, "parameters": fit["parameters"],
            "train_loss": fit["train_objective"], "validation_loss": fit["validation_objective"],
            **cap_methods[method],
        })
    diagnostic = [dict(p) for p in capacity]
    for fit in continuations["results"]:
        if fit["trial"]["distinct_examples"] != 2048:
            continue
        for path, digest in fit["source_sha256"].items():
            evidence.read(BASE / path, digest, jsonl=path.endswith(".jsonl"))
        family = fit["trial"]["architecture"]
        initial = next(p for p in capacity if p["family"] == family and p["width"] == 512)
        last = next(v for v in fit["validation_trajectory"] if v["step"] == 32768)
        method = initial["method"] + "_continue32768"
        diagnostic.append({
            **initial, **cap_methods[method], "method": method, "updates": 32768,
            "train_loss": last["groups"]["train"]["objective"],
            "validation_loss": last["groups"]["validation"]["objective"],
        })
    assert len(capacity) == 7 and len(diagnostic) == 9
    return data, capacity, diagnostic


def save(fig, folder, name):
    for suffix in ["pdf", "png"]:
        opts = {"metadata": {"CreationDate": None, "ModDate": None}} if suffix == "pdf" else {"dpi": 240}
        fig.savefig(folder / f"{name}.{suffix}", **opts)
    plt.close(fig)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9DFE5", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5)


def family_legend(ax, **kwargs):
    handles = [Line2D([], [], color=COLORS[f], marker=MARKERS[f], linestyle="none", markersize=5, label=LABELS[f]) for f in COLORS]
    return ax.legend(handles=handles, frameon=False, handletextpad=.4, borderaxespad=.3, **kwargs)


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    evidence = Evidence(root)
    evidence.inputs[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data, capacity, diagnostic = load_points(evidence)
    figures, generated = output / "figures", output / "generated"
    figures.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.4, "axes.labelsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.75), layout="constrained")
    ax = axes[0]
    xs = [p["records"] for p in data]
    ys = [p["throughput_ratio"] for p in data]
    err = [[p["throughput_ratio"]-p["throughput_ci95"][0] for p in data], [p["throughput_ci95"][1]-p["throughput_ratio"] for p in data]]
    ax.axvspan(512, 40000, color=COLORS["dense"], alpha=.055, linewidth=0)
    ax.errorbar(xs, ys, yerr=err, marker="o", markersize=3.9, color=COLORS["dense"], linewidth=1.4, capsize=2.2)
    ax.set_xscale("log", base=2)
    ax.set_xlim(11, 44000)
    ax.set_xticks([16, 64, 256, 512, 2048, 8192, 32768], ["16", "64", "256", "512", "2k", "8k", "32k"], rotation=35)
    ax.set_ylim(2.2, 5.85)
    ax.set_xlabel("Distinct training records")
    ax.set_ylabel("End-to-end throughput / AR")
    ax.set_title("(a) Data: 8,192 updates per fit", loc="left", pad=8)
    retention = next(p["tokens_per_second"] for p in data if p["records"] == 512) / max(p["tokens_per_second"] for p in data)
    ax.annotate(f"512 records retain {100*retention:.1f}%\nof the observed best", xy=(512, next(p["throughput_ratio"] for p in data if p["records"] == 512)), xytext=(120, 3.0), fontsize=8.5, arrowprops={"arrowstyle": "-", "color": "#68737D", "lw": .7})
    style(ax)
    ax = axes[1]
    for family in ["factorized", "mlp"]:
        pts = sorted([p for p in capacity if p["family"] == family and p["records"] == 2048], key=lambda p: p["parameters"])
        ax.plot([p["parameters"]/1e6 for p in pts], [p["throughput_ratio"] for p in pts], color=COLORS[family], linestyle="--" if family == "mlp" else "-", linewidth=1, alpha=.8)
    for p in capacity:
        y = p["throughput_ratio"]
        ax.errorbar(p["parameters"]/1e6, y, yerr=[[y-p["throughput_ci95"][0]], [p["throughput_ci95"][1]-y]], marker=MARKERS[p["family"]], markersize=5.4, markerfacecolor="white" if p["records"] == 512 else COLORS[p["family"]], color=COLORS[p["family"]], capsize=2.2, linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([11.79648, 23.59296, 52.4288, 94.37184], ["11.8", "23.6", "52.4", "94.4"])
    ax.set_xlim(9, 125)
    ax.set_ylim(3.65, 5.7)
    ax.set_xlabel("Mapper parameters (millions)")
    ax.set_ylabel("End-to-end throughput / AR")
    ax.set_title("(b) Capacity: 8,192 updates per fit", loc="left", pad=8)
    family_legend(ax, loc="lower right", fontsize=8.2)
    ax.text(.03, .98, "Open: 512 records\nFilled: 2,048 records", transform=ax.transAxes, va="top", fontsize=8)
    style(ax)
    save(fig, figures, "visual_scaling_main")

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.9), layout="constrained", sharey=True)
    for ax, split, letter in zip(axes, ["train", "validation"], ["a", "b"], strict=True):
        key = split + "_loss"
        for p in diagnostic:
            marker = "X" if p["updates"] == 32768 else MARKERS[p["family"]]
            ax.scatter(p[key], p["tokens_per_second"], color=COLORS[p["family"]], marker=marker, facecolors="white" if p["records"] == 512 else COLORS[p["family"]], s=32, zorder=3)
        for family in ["factorized", "mlp"]:
            pair = sorted([p for p in diagnostic if p["family"] == family and p["width"] == 512], key=lambda p:p["updates"])
            assert len(pair) == 2
            ax.annotate("", xy=(pair[1][key], pair[1]["tokens_per_second"]), xytext=(pair[0][key], pair[0]["tokens_per_second"]), arrowprops={"arrowstyle": "->", "color": COLORS[family], "lw": 1.1})
        dense = next(p for p in diagnostic if p["family"] == "dense" and p["records"] == 2048)
        mlp = next(p for p in diagnostic if p["family"] == "mlp" and p["width"] == 4096)
        ax.annotate("Dense, 2k records", xy=(dense[key], dense["tokens_per_second"]), xytext=(6, 9), textcoords="offset points", fontsize=8, color=COLORS["dense"])
        ax.annotate("MLP-4096, 2k", xy=(mlp[key], mlp["tokens_per_second"]), xytext=(5, -13), textcoords="offset points", fontsize=8, color=COLORS["mlp"])
        ax.set_xlabel(f"{split.capitalize()} relative interface MSE")
        ax.set_title(f"({letter}) {split.capitalize()} fit versus throughput", loc="left", pad=8)
        ax.set_xlim(.13 if split == "train" else .17, .307)
        ax.set_ylim(145, 211)
        ax.text(.03, .025, "Arrows / ×: 8,192 → 32,768 updates", transform=ax.transAxes, fontsize=8)
        style(ax)
    axes[0].set_ylabel("End-to-end throughput (tokens/s)")
    family_legend(axes[1], loc="center right", fontsize=8)
    save(fig, figures, "visual_scaling_loss_speed")

    result = {
        "status": "complete", "input_sha256": evidence.inputs,
        "data_scaling": data, "capacity": capacity, "loss_speed": diagnostic,
        "descriptive_correlations": {split: float(np.corrcoef([p[split+"_loss"] for p in diagnostic], [p["tokens_per_second"] for p in diagnostic])[0, 1]) for split in ["train", "validation"]},
        "scope": "DFlash-8B, NuminaMath fitting, one fitting seed (1729). Each evaluation uses 128 exposed MATH development requests and a 2048-output-token cap. Data scaling is one shared-control campaign; capacity/loss-speed are a separate shared-control campaign. Throughput is total generated tokens / summed request_seconds, including prefill. Data/capacity error bars are existing paired 95% request-bootstrap intervals; they exclude fitting-seed, repeated-run and selection uncertainty. The seven main capacity cells are all 8192-update candidates in the completed long-output evaluation, not all 30 capacity fits. Diagnostic scatter adds two 32768-update continuations. Correlations are descriptive across dependent, selected recipes, not causal or inferential. Filled markers:2048 records; open:512; X:continued fit. No exact-to-AR claim is made for these inherited BF16 runs.",
        "retained_existing_figures": ["numina_capacity.pdf", "numina_epochs.pdf", "numina_regularization.pdf", "target14_dflash_fitting.pdf", "target14_eagle3_fitting.pdf", "eagle_small_epochs.pdf"],
        "outputs": [f"figures/visual_scaling_{name}.{suffix}" for name in ["main", "loss_speed"] for suffix in ["pdf", "png"]],
    }
    (generated / "scaling_visuals_evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("paper/iclr2027"))
    args = parser.parse_args()
    result = build(args.root, args.output)
    print(json.dumps({"status": result["status"], "inputs": len(result["input_sha256"]), "outputs": result["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
