"""Build auditable exploratory research plots, separate from confirmatory paper claims."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("reports/autoresearch-20260907")
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
inputs = {}


def read(lane):
    p = ROOT / "run-28400" / f"lane{lane}" / "research-result.json"
    inputs[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    result = json.loads(p.read_text())
    if result["status"] != "pass" or result["summary"]["requests"] != 32:
        raise ValueError(f"Incomplete screen: {p}")
    return result["summary"]["methods"]


fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
rows = []
for i, (label, lane, method, fraction) in enumerate(
    [
        ("DFlash / GSM8K / two taps", 0, "relay_last2", 0.4),
        ("DFlash / GSM8K / rank 1024", 0, "relay_factor1024", 0.45),
        ("DFlash / MATH / two taps", 1, "relay_last2", 0.4),
        ("DFlash / MATH / rank 1024", 1, "relay_factor1024", 0.45),
        ("EAGLE-3 / GSM8K / two taps", 2, "relay_last2", 0.4),
    ]
):
    r = read(lane)[method]
    ratio = r["throughput_ratio"]
    low, high = r["throughput_ci95"]
    axes[0].errorbar(
        ratio * 100,
        i,
        xerr=[[100 * (ratio - low)], [100 * (high - ratio)]],
        fmt="o",
        capsize=3,
    )
    rows.append(dict(label=label, mapper_parameter_fraction=fraction, **r))
axes[0].set_yticks(range(5), [r["label"] for r in rows])
axes[0].invert_yaxis()
axes[0].set_title("Retargeted interfaces")
for rank in [1024, 1536, 2048]:
    r = read(3)[f"relay_svd{rank}"]
    ratio = r["throughput_ratio"]
    low, high = r["throughput_ci95"]
    fraction = rank * (20480 + 4096) / (20480 * 4096)
    axes[1].errorbar(
        fraction * 100,
        ratio * 100,
        yerr=[[100 * (ratio - low)], [100 * (high - ratio)]],
        fmt="o",
        capsize=3,
    )
    axes[1].annotate(
        str(rank),
        (fraction * 100, ratio * 100),
        xytext=(5, 3),
        textcoords="offset points",
        fontsize=9,
    )
    rows.append(
        dict(
            label=f"Native DFlash rank {rank}", mapper_parameter_fraction=fraction, **r
        )
    )
axes[1].set_xlabel("Native projection parameters retained (%)")
axes[1].set_ylabel("End-to-end throughput retained (%)")
axes[1].set_title("Native drafter projection, no fitting")
axes[0].set_xlabel("End-to-end throughput retained (%)")
axes[0].axvline(100, color="gray", linestyle="--", linewidth=1)
axes[1].axhline(100, color="gray", linestyle="--", linewidth=1)
for ax in axes:
    ax.grid(alpha=0.2)
fig.suptitle(
    "Exploratory screens: 32 development questions, 128-token output cap", fontsize=11
)
fig.tight_layout()
for suffix in ["png", "pdf"]:
    fig.savefig(OUT / f"interface_compression.{suffix}", dpi=200, bbox_inches="tight")
(ROOT / "compression-screen-summary.json").write_text(
    json.dumps(
        dict(
            inputs=inputs,
            rows=rows,
            scope="Exploratory paired request intervals; exposed development data; excludes fitting-seed and selection uncertainty. Parameter fractions concern the mapper/projection, not the full model. Native ratios use repacked native FC as reference.",
        ),
        indent=2,
    )
    + "\n"
)
