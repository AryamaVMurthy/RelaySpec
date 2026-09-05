#!/usr/bin/env python3
"""Build the figures introduced in the RelaySpec v2 manuscript.

Every number here is copied from a recorded run and is traceable to an
artifact named in docs/research/relayspec-decision-register.md or to
paper/iclr2027/generated/results_macros.tex. Nothing is estimated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PDF_METADATA = {"Creator": "", "Producer": "", "CreationDate": None}

WORKS = "#2c7d59"
PARTIAL = "#c78a1e"
FAILS = "#a6362f"
NEUTRAL = "#4878a8"


def _style() -> None:
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


def _save(fig: Any, path: Path) -> None:
    fig.savefig(path, format="pdf", bbox_inches="tight", metadata=PDF_METADATA)
    plt.close(fig)
    print(f"wrote {path}")


def adaptation_cost(path: Path, root: Path) -> None:
    """What it costs to support one new target, and what that costs you."""
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1.05, 1]}
    )

    # Panel (a): training examples processed to support one new target.
    labels = [
        "RelaySpec\nlinear map",
        "EAGLE-3\ndraft model",
        "PARD\ndraft model",
        "DFlash\ndraft model",
    ]
    examples = [4_096, 532_000, 4_000_000, 4_800_000]
    colors = [WORKS, "#8c8c8c", "#8c8c8c", "#8c8c8c"]
    positions = range(len(labels))
    left.barh(
        list(positions), examples, color=colors, edgecolor="#333333", linewidth=0.6
    )
    left.set_xscale("log")
    left.set_yticks(list(positions))
    left.set_yticklabels(labels)
    left.set_xlabel("Training examples processed for one target (log scale)")
    left.set_xlim(1e3, 3e7)
    left.grid(axis="x", color="#dddddd", linewidth=0.5, zorder=0)
    left.set_axisbelow(True)
    for index, value in enumerate(examples):
        text = (
            f"{value / 1e6:.1f}M"
            if value >= 1e6
            else (f"{value / 1e3:.0f}K" if value >= 1e4 else f"{value:,}")
        )
        if index == 0:
            text = f"{value:,}  (1,172x less than DFlash)"
        left.text(value * 1.4, index, text, va="center", fontsize=7.0)
    left.set_title("(a) Cost of supporting a new target")

    # Panel (b): throughput kept relative to a proposer trained for that
    # target, computed from the same paired-AR artifact the main table and
    # the results macros are generated from, so the three cannot disagree.
    import json

    main = json.loads(
        (root / "reports/final/MAIN_PAIRED_AR.json").read_text(encoding="utf-8")
    )
    label = {
        ("dflash", "8b"): "DFlash\n8B",
        ("eagle3", "8b"): "EAGLE-3\n8B",
        ("eagle3", "14b"): "EAGLE-3\n14B",
    }
    configs: list[str] = []
    recovery: list[float] = []
    for pair in main["pairs"]:
        specific = pair["target_specific_vs_ar"]
        if specific is None:
            continue
        configs.append(label[(pair["family"], pair["target"])])
        recovery.append(100.0 * pair["relay_vs_ar"]["estimate"] / specific["estimate"])
    bars = right.bar(
        configs, recovery, color=NEUTRAL, edgecolor="#333333", linewidth=0.6, width=0.55
    )
    right.axhline(100.0, color="#333333", linestyle="--", linewidth=0.8)
    right.text(
        -0.42,
        103.0,
        "a proposer trained for this target",
        fontsize=6.6,
        ha="left",
        color="#333333",
    )
    right.set_ylim(0, 122)
    right.set_ylabel("Throughput kept (%)")
    right.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
    right.set_axisbelow(True)
    for bar, value in zip(bars, recovery):
        right.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:.1f}",
            ha="center",
            fontsize=7.0,
        )
    right.set_title("(b) What the cheaper map keeps")
    _save(fig, path)


def transfer_speedups(path: Path, root: Path) -> None:
    """Speedup over plain autoregressive decoding for every retargeting setting.

    Reads the same two artifacts the tables are generated from, so a figure
    and a table can never disagree about the same number.
    """
    import json

    main = json.loads(
        (root / "reports/final/MAIN_PAIRED_AR.json").read_text(encoding="utf-8")
    )
    transfer = json.loads(
        (root / "reports/final/TRANSFER_PAIRED_AR.json").read_text(encoding="utf-8")
    )

    # Each group carries TWO encodings, colour and hatch, so the figure
    # stays readable in greyscale and for colourblind readers rather than
    # depending on the green/blue distinction alone.
    rows: list[tuple[str, float, float, float, str, str]] = []
    label = {
        ("dflash", "8b"): "Same family, larger target\n(Qwen3-4B proposer to Qwen3-8B)",
        (
            "dflash",
            "14b",
        ): "Same family, larger target\n(Qwen3-4B proposer to Qwen3-14B)",
    }
    for pair in main["pairs"]:
        key = (pair["family"], pair["target"])
        if key not in label:
            continue
        r = pair["relay_vs_ar"]
        rows.append((label[key], r["estimate"], r["lower"], r["upper"], WORKS, "//"))
    short = {
        "fine_tuned_descendant": "Independently fine-tuned\ndescendant of the target",
        "adapter_small_to_large": "Frozen decoder plus adapter\n(8B map serving a 14B target)",
        "cross_family_smaller_target": "Different family, smaller target\n(Llama-3.1-8B proposer to Llama-3.2-3B)",
        "cross_tokenizer": "Different tokenizer\n(Qwen3 proposer to Llama-3.1-8B)",
    }
    for setting in transfer["settings"]:
        r = setting["relay_vs_ar"]
        rows.append(
            (short[setting["key"]], r["estimate"], r["lower"], r["upper"], NEUTRAL, "")
        )

    rows.sort(key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    positions = list(range(len(rows)))
    for pos, (_, value, low, high, color, hatch) in zip(positions, rows):
        ax.barh(
            pos,
            value,
            color=color,
            edgecolor="#333333",
            linewidth=0.6,
            height=0.6,
            hatch=hatch,
        )
        ax.errorbar(
            value,
            pos,
            xerr=[[value - low], [high - value]],
            fmt="none",
            ecolor="#222222",
            elinewidth=0.9,
            capsize=2.5,
        )
        ax.text(high + 0.12, pos, f"{value:.2f}x", va="center", fontsize=7.2)
    ax.set_yticks(positions)
    ax.set_yticklabels([row[0] for row in rows])
    ax.axvline(1.0, color="#a6362f", linestyle="--", linewidth=1.0)
    ax.text(1.06, -0.75, "plain autoregressive decoding", fontsize=6.8, color="#a6362f")
    ax.set_xlim(0, 6.4)
    ax.set_ylim(-1.0, len(rows) - 0.35)
    ax.set_xlabel(
        "End-to-end speedup over plain autoregressive decoding (paired, 95% CI)"
    )
    ax.grid(axis="x", color="#dddddd", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    _save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    root = Path(__file__).resolve().parents[1]
    adaptation_cost(args.output_dir / "adaptation_cost.pdf", root)
    transfer_speedups(args.output_dir / "transfer_speedups.pdf", root)


if __name__ == "__main__":
    main()
