"""Rebuild transfer figures from checked, paired request records.

All speed ratios use total generated tokens / total request seconds, including
prefill. The two main panels deliberately retain different runtime/cohort labels
and references: matched FP32 AR for family transfer; matched native BF16 vLLM
for the independent rollout experiment. No cross-runtime ranking is implied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
import numpy as np

from build_family_extension_paper_assets import build as verify_extension


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#78838D"
SEED = 1729
RESAMPLES = 10000
FAMILIES = [("llama", "Llama-8B → Llama-3B"),
            ("cross", "Qwen-4B → Llama-8B")]
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": .65,
    "axes.edgecolor": "#57616A", "grid.color": "#DDE2E7",
    "grid.linewidth": .6, "pdf.fonttype": 42,
    "savefig.dpi": 200, "figure.facecolor": "white",
}


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ratio_ci(numerator, denominator, seed=SEED):
    a, b = np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    assert a.shape == b.shape and a.ndim == 1 and len(a) > 0
    assert np.all(a > 0) and np.all(b > 0)
    ids = np.random.default_rng(seed).integers(0, len(a), (RESAMPLES, len(a)))
    ratio = float(a.sum() / b.sum())
    ci = np.quantile(a[ids].sum(1) / b[ids].sum(1), [.025, .975])
    return {"ratio": ratio, "ci": ci.tolist()}


def _save(fig, folder, name):
    """Fixed canvas and metadata make same-input exports byte reproducible."""
    folder.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    for label in fig.findobj(Text):
        if label.get_visible() and label.get_text():
            box = label.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= width+1 and box.y1 <= height+1, (
                name, label.get_text(), box.bounds, (width, height))
    fig.savefig(folder / f"{name}.pdf", metadata={
        "Creator": "RelaySpec build_transfer_visuals.py",
        "CreationDate": None, "ModDate": None,
    })
    fig.savefig(folder / f"{name}.png", metadata={"Software": "RelaySpec"})
    plt.close(fig)


def _main_figure(evidence, folder):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.2, 2.65),
                                gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=.12, right=.975, bottom=.30, top=.73, wspace=.50)
    for fi, (family, _) in enumerate(FAMILIES):
        row = evidence["family128"][family]
        y = 1 - fi
        for method, shift, color, marker in [
                ("old", -.14, BLUE, "o"), ("selected", .14, ORANGE, "D")]:
            value = row[method]["request_tps"] / row["ar"]["request_tps"]
            lo, hi = row[method]["ar_ratio_ci"]
            ax.errorbar(value, y + shift, xerr=[[value-lo], [hi-value]],
                        fmt=marker, color=color, markersize=5, capsize=2,
                        linewidth=1.3, label="4,096-record baseline" if method == "old" and fi == 0
                        else "Selected larger fit" if fi == 0 else None)
            ax.text(value, y + shift + (.15 if method == "selected" else -.28),
                    f"{value:.2f}×", color=color, ha="center", va="bottom", fontsize=8)
    ax.axvline(1, color=GRAY, linestyle="--", linewidth=.9)
    ax.set(xlim=(.95, 3.15), ylim=(-.65, 1.65),
           yticks=[1, 0], yticklabels=["Llama →\nLlama", "Qwen →\nLlama"],
           xlabel="Relay / matched AR throughput")
    ax.set_xticks([1, 1.5, 2, 2.5, 3])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.grid(axis="x", alpha=.8)
    ax.set_axisbelow(True)
    ax.set_title("(a) Frozen-drafter portability", loc="left", pad=31,
                 fontweight="semibold")
    ax.text(0, 1.14, "MATH-500 · FP32 target · 128 per pair", transform=ax.transAxes,
            fontsize=8, color="#45515B")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.095, .13),
               frameon=False, ncol=1, handlelength=1.2, labelspacing=.25)

    for row, y, marker in zip(evidence["rollout_transfer"], [1, 0], ["o", "s"]):
        value, (lo, hi) = row["mapped_over_native"], row["ci"]
        bx.errorbar(value, y, xerr=[[value-lo], [hi-value]], fmt=marker,
                    color=GREEN, markersize=5, capsize=2, linewidth=1.4)
        bx.text(value, y+.2, f"+{100*(value-1):.2f}%", color=GREEN,
                ha="center", fontsize=8.5)
    bx.axvline(1, color=GRAY, linestyle="--", linewidth=.9)
    bx.set(xlim=(.997, 1.039), ylim=(-.5, 1.65),
           yticks=[1, 0], yticklabels=["Original\nGPUs", "Swapped\nGPUs"],
           xlabel="Relay / native throughput")
    bx.set_xticks([1, 1.01, 1.02, 1.03])
    bx.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}×"))
    bx.grid(axis="x", alpha=.8)
    bx.set_axisbelow(True)
    bx.set_title("(b) Rollout-calibrated reuse", loc="left", pad=31,
                 fontweight="semibold")
    bx.text(0, 1.14, "Numina · BF16 vLLM · 128 per run", transform=bx.transAxes,
            fontsize=8, color="#45515B")
    fig.text(.615, .06, "128/128 exact sequences\nin every relay arm", fontsize=8,
             color="#45515B", va="center")
    _save(fig, folder, "visual_transfer_main")


def _gain_figure(diagnostics, folder):
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.75), sharey=True)
    fig.subplots_adjust(left=.125, right=.985, bottom=.22, top=.79, wspace=.16)
    for ax, (family, label) in zip(axes, FAMILIES):
        data = diagnostics[family]
        pts = data["requests"]
        ax.scatter([r["output_tokens"] for r in pts],
                   [r["selected_over_old_request_speed"] for r in pts],
                   s=13, color=GRAY, alpha=.43, linewidths=0,
                   label="Individual requests", zorder=2)
        bins = data["length_bins"]
        for i, b in enumerate(bins):
            v, (lo, hi) = b["ratio"], b["ci"]
            ax.errorbar(b["median_output_tokens"], v,
                        yerr=[[v-lo], [hi-v]], fmt="D", markersize=4.5,
                        color=ORANGE, linewidth=1.3, capsize=2, zorder=3,
                        label="Within-bin aggregate + 95% CI" if i == 0 else None)
        ax.axhline(1, color="#434B52", linestyle="--", linewidth=.8, zorder=1)
        ax.set_xscale("log", base=2)
        ax.set(xlim=(64, 2500), ylim=(.75, 1.6), xticks=[128, 512, 2048],
               xlabel="Generated target tokens (cap 2,048)")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}×"))
        ax.grid(alpha=.65)
        ax.set_title(label, loc="left", fontweight="semibold")
        ax.text(.03, .94, f"Faster on {data['requests_faster']}/128 requests",
                transform=ax.transAxes, va="top", fontsize=8)
    axes[0].set_ylabel("Selected / baseline\nrequest speed")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, 1.005),
               frameon=False, ncol=2, columnspacing=1, handletextpad=.5)
    _save(fig, folder, "visual_transfer_request_gain")


def _verification_figure(diagnostics, folder):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.2, 2.8),
                                gridspec_kw={"width_ratios": [.9, 1.1]})
    fig.subplots_adjust(left=.12, right=.98, bottom=.22, top=.78, wspace=.36)
    for i, (family, _) in enumerate(FAMILIES):
        for method, shift, color, marker in [
                ("old", -.16, BLUE, "o"), ("selected", .16, ORANGE, "D")]:
            d = diagnostics[family]["verification"][method]
            v, (lo, hi) = d["ratio"], d["ci"]
            ax.errorbar(i+shift, v, yerr=[[v-lo], [hi-v]], fmt=marker,
                        color=color, markersize=5, capsize=2, linewidth=1.3,
                        label="4,096-record baseline" if i == 0 and method == "old"
                        else "Selected larger fit" if i == 0 else None)
            ax.text(i+shift, hi+.12, f"{v:.2f}", ha="center", color=color, fontsize=8)
        o, s = (diagnostics[family]["verification"][m]["target_calls"] for m in ["old", "selected"])
        ax.text(i, .19, f"{100*(s/o-1):+.1f}% calls", ha="center", fontsize=8)
    ax.set(xlim=(-.55, 1.55), ylim=(0, 4.55),
           xticks=[0, 1], xticklabels=["Llama → Llama", "Qwen → Llama"],
           ylabel="Output target tokens /\nverification call")
    ax.grid(axis="y", alpha=.65)
    ax.set_title("(a) Verification efficiency", loc="left", fontweight="semibold")
    for (family, label), color, marker in zip(FAMILIES, [BLUE, GREEN], ["o", "^"]):
        r = diagnostics[family]["requests"]
        bx.scatter([p["selected_over_old_verification_progress"] for p in r],
                   [p["selected_over_old_request_speed"] for p in r],
                   s=16, color=color, marker=marker, alpha=.5, linewidths=0,
                   label="Llama → Llama" if family == "llama" else "Qwen → Llama")
    bx.plot([.75, 1.65], [.75, 1.65], color=GRAY, linestyle=":", linewidth=.9)
    bx.axhline(1, color=GRAY, linestyle="--", linewidth=.65)
    bx.axvline(1, color=GRAY, linestyle="--", linewidth=.65)
    bx.set(xlim=(.75, 1.65), ylim=(.75, 1.65),
           xlabel="Selected / baseline verification progress",
           ylabel="Selected / baseline request speed")
    bx.set_xticks([.75, 1, 1.25, 1.5])
    bx.set_yticks([.75, 1, 1.25, 1.5])
    bx.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}×"))
    bx.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}×"))
    bx.grid(alpha=.5)
    bx.set_title("(b) Paired request changes", loc="left", fontweight="semibold")
    ax.legend(loc="upper left", bbox_to_anchor=(-.15, 1.42), frameon=False,
              fontsize=7.8, handlelength=1.1, labelspacing=.3)
    bx.legend(loc="upper left", bbox_to_anchor=(-.06, 1.42), frameon=False,
              fontsize=7.8, handlelength=1.1, labelspacing=.3)
    _save(fig, folder, "visual_transfer_verification")


def _length_figure(diagnostics, folder):
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.55), sharey=True)
    fig.subplots_adjust(left=.085, right=.975, bottom=.21, top=.77, wspace=.14)
    for ax, field, title in zip(axes, ["output_tokens", "input_tokens"],
                               ["(a) Observed output length", "(b) Prompt length"]):
        for (family, _), color, line in zip(FAMILIES, [BLUE, GREEN], ["-", "--"]):
            x = np.sort([r[field] for r in diagnostics[family]["requests"]])
            y = np.arange(1, len(x)+1)/len(x)
            ax.step(np.r_[0, x], np.r_[0, y], where="post", color=color,
                    linestyle=line, linewidth=1.7,
                    label="Llama → Llama" if family == "llama" else "Qwen → Llama")
        ax.set(ylim=(0, 1.025), xlabel="Target-tokenizer tokens")
        ax.grid(alpha=.6)
        ax.set_title(title, loc="left", fontweight="semibold")
    axes[0].set(xlim=(0, 2120), xticks=[0, 512, 1024, 1536, 2048],
                ylabel="Fraction of 128 requests")
    axes[0].axvline(2048, color=GRAY, linestyle=":", linewidth=1)
    axes[0].text(.45, .22, "At the 2,048-token cap:\nLlama: 8/128 · Qwen→Llama: 24/128",
                 transform=axes[0].transAxes, fontsize=7.7, ha="center", va="center")
    axes[1].set_xlim(left=0)
    axes[1].set_xticks([0, 50, 100, 150, 200, 250])
    axes[0].set_yticks([0, .2, .4, .6, .8, 1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(.5, .995))
    _save(fig, folder, "visual_transfer_lengths")


def build(root: Path, output: Path):
    root, output = Path(root).resolve(), Path(output).resolve()
    # The existing evidence builder checks configurations, revisions, actual
    # dtypes, source hashes, token hashes, complete 128-key coverage and literal
    # token-array equality. Its outputs go to a temporary directory so this
    # builder does not mutate tables or evidence owned by another build step.
    with tempfile.TemporaryDirectory(prefix="relayspec-transfer-audit-") as tmp:
        evidence = verify_extension(root, Path(tmp))
    existing = root / "paper/iclr2027/generated/family_extension_evidence.json"
    recorded = json.loads(existing.read_text())
    assert recorded == evidence, "Transfer source evidence is stale: rebuild its own assets first"
    inputs = dict(evidence["inputs"])
    inputs[str(existing.relative_to(root))] = _digest(existing)
    source = root / "scripts/build_transfer_visuals.py"
    inputs[str(source.relative_to(root))] = _digest(source)
    verifier = root / "scripts/build_family_extension_paper_assets.py"
    inputs[str(verifier.relative_to(root))] = _digest(verifier)

    groups = {(family, method): {} for family, _ in FAMILIES
              for method in ["ar", "old", "selected"]}
    for wave in range(16):
        for lane in range(4):
            path = root / f"reports/family-eval128-20260908/artifacts/wave{wave}/lane{lane}/shared/benchmark-rank0.jsonl"
            assert inputs[str(path.relative_to(root))] == _digest(path)
            family = "llama" if lane < 2 else "cross"
            for row in map(json.loads, path.read_text().splitlines()):
                method = {"native_ar": "ar", "relay_p": "old",
                          "relay_p_cross_family": "old",
                          "relay_candidate_1": "selected"}[row["method"]]
                groups[family, method][row["problem_id"]] = row

    diagnostics = {}
    for family, _ in FAMILIES:
        keys = sorted(groups[family, "ar"])
        assert len(keys) == 128
        rows = []
        for k in keys:
            ar, old, selected = (groups[family, m][k] for m in ["ar", "old", "selected"])
            assert ar["output_token_ids"] == old["output_token_ids"] == selected["output_token_ids"]
            assert ar["input_tokens"] == old["input_tokens"] == selected["input_tokens"]
            assert old["target_calls"] == len(old["acceptance_lengths"]) > 0
            assert selected["target_calls"] == len(selected["acceptance_lengths"]) > 0
            rows.append({"problem_id": k, "input_tokens": ar["input_tokens"],
                         "output_tokens": ar["output_tokens"],
                         "ar_request_seconds": ar["request_seconds"],
                         "old_request_seconds": old["request_seconds"],
                         "selected_request_seconds": selected["request_seconds"],
                         "old_target_calls": old["target_calls"],
                         "selected_target_calls": selected["target_calls"],
                         "selected_over_old_request_speed": old["request_seconds"] / selected["request_seconds"],
                         "selected_over_old_verification_progress": old["target_calls"] / selected["target_calls"]})
        bins = []
        for lower, upper in [(0, 256), (256, 512), (512, 1024), (1024, 2049)]:
            selected = [r for r in rows if lower <= r["output_tokens"] < upper]
            assert selected
            bins.append({"lower_inclusive": lower, "upper_exclusive": upper,
                         "requests": len(selected),
                         "median_output_tokens": float(np.median([r["output_tokens"] for r in selected])),
                         **_ratio_ci([r["old_request_seconds"] for r in selected],
                                     [r["selected_request_seconds"] for r in selected])})
        verification = {}
        for method in ["old", "selected"]:
            verification[method] = {
                **_ratio_ci([r["output_tokens"] for r in rows], [r[f"{method}_target_calls"] for r in rows]),
                "output_tokens": sum(r["output_tokens"] for r in rows),
                "target_calls": sum(r[f"{method}_target_calls"] for r in rows),
            }
        diagnostics[family] = {"requests": rows, "length_bins": bins,
                               "verification": verification,
                               "requests_faster": sum(r["selected_over_old_request_speed"] > 1 for r in rows),
                               "cap_reached": sum(r["output_tokens"] == 2048 for r in rows)}

    folder = output / "figures"
    with plt.rc_context(STYLE):
        _main_figure(evidence, folder)
        _gain_figure(diagnostics, folder)
        _verification_figure(diagnostics, folder)
        _length_figure(diagnostics, folder)
    result = {
        "inputs_sha256": inputs, "family128": evidence["family128"],
        "rollout_transfer": evidence["rollout_transfer"], "diagnostics": diagnostics,
        "definitions": {
            "request_throughput": "sum generated target tokens / sum request_seconds; includes prefill and generation, excludes loading/tokenization/final text decoding",
            "verification_progress": "sum final output target tokens / sum target_calls; calls count verification-loop forwards and exclude prefill. Final output counts are EOS/cap-trimmed and include the prefill's first token. This is an effective output-per-call measure, not proposal acceptance fraction.",
            "paired_request_gain": "old_request_seconds / selected_request_seconds for identical output token arrays",
            "length_bin_gain": "sum old request time / sum selected request time within each fixed output-length bin; points at median observed output length",
            "uncertainty": "95% percentile intervals from 10,000 paired request resamples; seed 1729 for family/diagnostic plots, seed 42 for rollout repeats. Conditional on selected checkpoints and runtime; excludes fitting-seed, selection and repeat-to-repeat uncertainty.",
            "length_ecdf": "128 observed lengths per model pair, with outputs right-censored at 2,048 generated tokens. Input strings share questions but rendered/tokenized lengths can differ by target.",
            "comparison_scope": "The MATH-500 FP32/Hugging Face and Numina BF16/vLLM panels have different cohorts, targets, runtimes and references; they are not ranked against each other.",
            "selection": "Baseline is historical 4,096-record fit; selected is 16,384-record dense epoch 6 of a 12-epoch schedule for Llama and 8,192-record dense epoch 3 of a 24-epoch schedule for Qwen→Llama. Recipe changes confound a pure data-count attribution.",
        },
        "figures": {p.name: _digest(p) for p in sorted(folder.glob("visual_transfer*"))},
    }
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "transfer_visuals_evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("paper/iclr2027"))
    args = parser.parse_args()
    result = build(args.root, args.output)
    print(json.dumps({"figures": list(result["figures"]),
                      "input_files": len(result["inputs_sha256"]),
                      "requests_faster": {f: d["requests_faster"]
                                          for f, d in result["diagnostics"].items()}}, indent=2))
