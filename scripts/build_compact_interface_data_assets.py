"""Combine two fixed fitting seeds without treating them as new test questions."""

import json
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

analyze = runpy.run_path("scripts/analyze_compact_interface_data.py")["analyze"]
root = Path("reports/autoresearch-20260907")
waves = [
    json.loads(Path(f"configs/autoresearch/20260907/wave{n}.json").read_text())
    for n in [40, 41]
]
for first, second in zip(waves[0]["lanes"], waves[1]["lanes"]):
    for key in [
        "checkpoint",
        "checkpoint_sha256",
        "config",
        "feature_cache",
        "pilot_gate",
        "native_teacher",
        "requests",
        "max_new_tokens",
        "include_ar",
        "include_native",
    ]:
        if first[key] != second[key]:
            raise ValueError(f"Seed replication changed {key}")
    if {k: v for k, v in first["trial"].items() if k not in ["seed", "name"]} != {
        k: v for k, v in second["trial"].items() if k not in ["seed", "name"]
    }:
        raise ValueError("Seed replication changed fitting settings")
combined = []
for job, wave, name in [
    (28530, 40, "compact-interface-data-summary.json"),
    (28531, 41, "compact-interface-data-seed1730-summary.json"),
]:
    result = analyze(job, wave, root / name)
    combined.extend(result["results"])
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for role, color in [("native", "#245b87"), ("retargeted", "#b36b2b")]:
    for seed, marker, linestyle in [(1729, "o", "-"), (1730, "s", "--")]:
        cells = sorted(
            [
                r
                for r in combined
                if r["role"] == role and r["fit"]["trial"]["seed"] == seed
            ],
            key=lambda r: r["records"],
        )
        m = [r["decoding"]["methods"]["relay_reduced"] for r in cells]
        x = [r["records"] for r in cells]
        y = [v["throughput_ratio"] for v in m]
        ax.errorbar(
            x,
            y,
            yerr=[
                [v["throughput_ratio"] - v["throughput_ci95"][0] for v in m],
                [v["throughput_ci95"][1] - v["throughput_ratio"] for v in m],
            ],
            color=color,
            marker=marker,
            linestyle=linestyle,
            capsize=3,
            label=f"{role.title()}, seed {seed}",
        )
ax.axhline(1, color="black", linewidth=1, linestyle=":")
ax.set_xscale("log", base=2)
ax.set_xticks([16, 128], ["16", "128"])
ax.set_xlabel("Distinct calibration records (8,192 updates)")
ax.set_ylabel("Throughput / own 512-record two-layer map")
ax.grid(alpha=0.2)
ax.legend(fontsize=8)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(root / f"compact-interface-data.{ext}", dpi=180)
lines = [
    "| Role | Seed | Records | Train error | Validation error | Retention (95% request CI) |",
    "|---|---:|---:|---:|---:|---:|",
]
for r in combined:
    m = r["decoding"]["methods"]["relay_reduced"]
    lo, hi = m["throughput_ci95"]
    f = r["fit"]
    lines.append(
        f"| {r['role']} | {f['trial']['seed']} | {r['records']} | {f['train_objective']:.4f} | {f['validation_objective']:.4f} | {100 * m['throughput_ratio']:.1f}% [{100 * lo:.1f},{100 * hi:.1f}] |"
    )
(root / "compact-interface-data.md").write_text(
    "\n".join(lines)
    + "\n\nEight shared exposed questions. Intervals resample requests within each seed; seeds do not add independent questions. Original role-specific teacher, output width and normalization are retained. References are the original512-record two-layer maps. Feature errors are not comparable across roles.\n"
)


def span(role, n):
    values = [
        100 * r["decoding"]["methods"]["relay_reduced"]["throughput_ratio"]
        for r in combined
        if r["role"] == role and r["records"] == n
    ]
    if len(values) != 2:
        raise ValueError("Incomplete seed range")
    return f"{min(values):.1f}--{max(values):.1f}\\%"


text = r"""\paragraph{Calibration data depends on the interface role.}
We fit native and retargeted two-layer maps on the same 16- or 128-record
prefixes, at 8,192 batch-four updates and two fitting seeds. Each map is
compared with its role's original 512-record two-layer reference on the
same eight exposed GSM8K questions, with a 512-token cap.
"""
text += f"At 128 records, native maps retain {span('native', 128)} throughput across seeds, versus {span('retargeted', 128)} for retargeting. At 16 records these ranges fall to {span('native', 16)} and {span('retargeted', 16)}, respectively.\n"
text += r"""The 16-record fits closely match training features but have much larger
validation errors. These standard recipes retain different teacher outputs,
output widths and normalization, so the contrast does not isolate an
intrinsic difference in sample complexity. The two seeds share questions
and reference maps. This is a calibration-sensitivity screen, not a
minimum-data guarantee or independent quality confirmation.
"""
Path("paper/iclr2027/generated/compact_interface_data_paragraph.tex").write_text(text)
