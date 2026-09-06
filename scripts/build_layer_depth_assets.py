"""Compare equal-capacity tap choices on exactly shared development requests."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from relayspec.ar_paper_evidence import summarize

root = Path("reports/autoresearch-20260907")
inputs = {}


def rows(job, lane):
    p = root / f"run-{job}" / f"lane{lane}" / "benchmark-rank0.jsonl"
    inputs[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return [json.loads(x) for x in p.read_text().splitlines()]


fig, ax = plt.subplots(figsize=(6.4, 3.4))
results = []
for family, lanes, late_lane, color, offset in [
    ("DFlash", [0, 1], 0, "#286394", -0.12),
    ("EAGLE-3", [2, 3], 2, "#bb6230", 0.12),
]:
    selected = None
    for position, lane in enumerate(lanes):
        r = rows(28421, lane)
        ids = {x["problem_id"] for x in r}
        if selected is not None and ids != selected:
            raise ValueError("Depth screens use different questions")
        selected = ids
        report = summarize(r, reference="relay_base")["methods"]["relay_reduced"]
        results.append(
            dict(
                family=family,
                taps=[[1, 9], [9, 25]][position],
                requests=len(ids),
                **report,
            )
        )
    r = [
        x
        for x in rows(28400, late_lane)
        if x["problem_id"] in selected and x["method"] in ["relay_base", "relay_last2"]
    ]
    if {x["problem_id"] for x in r} != selected:
        raise ValueError("Late-layer screen missing matching questions")
    report = summarize(r, reference="relay_base")["methods"]["relay_last2"]
    results.append(dict(family=family, taps=[25, 33], requests=len(selected), **report))
    for position, result in enumerate(results[-3:]):
        v = result["throughput_ratio"]
        lo, hi = result["throughput_ci95"]
        ax.errorbar(
            position + offset,
            v * 100,
            yerr=[[100 * (v - lo)], [100 * (hi - v)]],
            fmt="o" if family == "DFlash" else "s",
            color=color,
            capsize=4,
            label=family if position == 0 else None,
        )
ax.set_xticks(range(3), ["Early: 1, 9", "Spaced: 9, 25", "Late: 25, 33"])
ax.set_ylabel("Throughput retained vs. full mapper (%)")
ax.set_xlabel("Two target layers; equal mapper parameter count")
ax.axhline(100, color="gray", linestyle="--", linewidth=1)
ax.grid(axis="y", alpha=0.2)
ax.legend(loc="lower right")
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(Path("paper/iclr2027/figures") / f"layer_depth.{ext}", dpi=200)
(root / "layer-depth-summary.json").write_text(
    json.dumps(
        dict(
            inputs=inputs,
            results=results,
            scope="Same16 exposed GSM8K questions,128-token cap,512 fitting records,8192updates,seed1729. Each candidate is paired with its own concurrent full-map baseline. Late-arm rows are the matching subset of the earlier32-request screen. Request bootstrap excludes fitting-seed and selection uncertainty.",
        ),
        indent=2,
    )
    + "\n"
)
print(
    [(r["family"], r["taps"], round(100 * r["throughput_ratio"], 2)) for r in results]
)
