"""Separate nested native CUDA-event spans without double-counting the heads."""
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


root = Path("reports")
records = []
for path in sorted(root.glob("run-*/lane[0-3]/profile-result.json")):
    data = json.loads(path.read_text())
    if data["status"] == "pass":
        records.extend({**r, "source": str(path)} for r in data["rows"])
groups = defaultdict(lambda: defaultdict(float))
for row in records:
    phases = row["phases"]
    p = lambda name: phases.get(name, {}).get("seconds", 0.)
    parts = {
        "Target backbone": p("verification")-p("verification_head"),
        "Target vocabulary head": p("verification_head"),
        "Draft backbone": p("draft"),
        "Draft vocabulary head": p("draft_head"),
        "Prefill": p("prefill"),
    }
    parts["Other / unassigned"] = row["wall_seconds"]-sum(parts.values())
    for benchmark in ["all", row["benchmark"]]:
        groups[benchmark]["wall_seconds"] += row["wall_seconds"]
        for name, value in parts.items():
            groups[benchmark][name] += value
summary = {benchmark: {"wall_seconds": value["wall_seconds"], "seconds": {k: v for k, v in value.items() if k != "wall_seconds"}, "fractions": {k: v/value["wall_seconds"] for k, v in value.items() if k != "wall_seconds"}} for benchmark, value in groups.items()}
(root/"native-profile-summary.json").write_text(json.dumps(summary, indent=2)+"\n")
lines = ["# Native generation time attribution", "", "Eight development requests, two per workload, at a512-token output cap. CUDA-event instrumentation exactly reproduced all native tokens and acceptance lengths. Event spans may include GPU idle gaps during host dispatch; instrumentation adds overhead. These are diagnostic fractions, not kernel-only timings or candidate speed measurements. Target vocabulary-head spans are subtracted from target verification before summing components.", "", "| Component | Seconds across requests | Fraction of instrumented wall time |", "|---|---:|---:|"]
for name, value in summary.get("all", {}).get("seconds", {}).items():
    lines.append(f"| {name} | {value:.3f} | {summary['all']['fractions'][name]:.1%} |")
if summary:
    frac = summary["all"]["fractions"]["Target vocabulary head"]
    lines += ["", f"Using these diagnostic spans as an Amdahl-style estimate, removing all target vocabulary-head time would yield only about{1/(1-frac):.3f}× throughput if all other work stayed fixed. Lazy projection also adds synchronization. This supports deprioritizing head-only verification optimizations for the10% objective and focusing on accepted progress per expensive target pass."]
(root/"NATIVE_PROFILE.md").write_text("\n".join(lines)+"\n")
if summary:
    benchmarks = list(summary)
    fig, ax = plt.subplots(figsize=(9, 4.4), layout="constrained")
    left = np.zeros(len(benchmarks))
    for name in summary["all"]["fractions"]:
        values = np.array([summary[b]["fractions"][name]*100 for b in benchmarks])
        ax.barh(benchmarks, values, left=left, label=name)
        left += values
    ax.set_xlabel("Instrumented generation wall time (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncols=3, fontsize=8)
    fig.savefig(root/"native-profile.png", dpi=180)
    fig.savefig(root/"native-profile.pdf")
print(json.dumps(summary.get("all", {})))
