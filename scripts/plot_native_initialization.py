"""Plot all matched inherited/random budgets without promoting repeat runs to seeds."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("reports/autoresearch-20260907")
cells = []
for name in [
    "native-initialization-summary",
    "native-initialization-budget-summary",
    "native-short-controls-summary",
]:
    cells.extend(json.loads((root / f"{name}.json").read_text())["results"])
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, n in zip(axes, [16, 128]):
    for init, color, marker in [
        ("random", "#245b87", "o"),
        ("native_columns", "#b36b2b", "s"),
    ]:
        selected = sorted(
            [
                c
                for c in cells
                if c["records"] == n
                and c["fit"]["trial"]["initialization"] == init
                and c["fit"]["trial"]["seed"] == 1729
            ],
            key=lambda c: c["fit"]["trial"]["steps"],
        )
        x = [c["fit"]["trial"]["steps"] for c in selected]
        ms = [c["decoding"]["methods"]["relay_reduced"] for c in selected]
        y = [m["throughput_ratio"] for m in ms]
        ax.errorbar(
            x,
            y,
            yerr=[
                [m["throughput_ratio"] - m["throughput_ci95"][0] for m in ms],
                [m["throughput_ci95"][1] - m["throughput_ratio"] for m in ms],
            ],
            color=color,
            marker=marker,
            capsize=3,
            label="Inherited columns"
            if init == "native_columns"
            else "Random initialization",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([128, 1024, 8192], ["128", "1,024", "8,192"])
    ax.set_xlabel("Mapper updates")
    ax.set_title(f"{n} calibration records")
    ax.axhline(1, color="black", ls=":", lw=1)
    ax.grid(alpha=0.2)
axes[0].set_ylabel("Throughput /512-record compact reference")
axes[1].legend(fontsize=8)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(root / f"native-initialization-budget.{ext}", dpi=180)
