"""Report all correction-head fits, including regressions and protocol changes."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


root = Path("reports")
rows = []
for path in sorted(root.glob("run-*/lane[0-3]/head-result.json")):
    result = json.loads(path.read_text())
    if result["status"] != "pass":
        continue
    raw = [json.loads(line) for line in (path.parent/"evaluation.jsonl").read_text().splitlines()]
    request_ids = sorted({r["problem_id"] for r in raw})
    matrices = []
    for method in ["native", "candidate"]:
        group = [r for r in raw if r["method"] == method]
        matrices.append(np.asarray([[sum(r["output_tokens"] for r in group if r["problem_id"] == problem), sum(r["seconds"] for r in group if r["problem_id"] == problem)] for problem in request_ids]))
    base, arm = matrices
    draws = np.random.default_rng(1729).integers(len(request_ids), size=(10000, len(request_ids)))
    a, b = arm[draws].sum(1), base[draws].sum(1)
    boot = (a[:, 0]/a[:, 1])/(b[:, 0]/b[:, 1])
    final = result["validation"][-1]
    workload_ratios = {}
    for benchmark in result["config"]["benchmarks"]:
        speeds = {}
        for method in ["native", "candidate"]:
            group = [r for r in raw if r["method"] == method and r["benchmark"] == benchmark]
            speeds[method] = sum(r["output_tokens"] for r in group)/sum(r["seconds"] for r in group)
        workload_ratios[benchmark] = speeds["candidate"]/speeds["native"]
    rows.append({
        "run": path.parents[1].name, "lane": path.parent.name,
        "name": result["config"]["name"], "config": result["config"],
        "counts": result["counts"], "summary": result["summary"],
        "native_ratio": result["native_ratio"], "ci95": np.quantile(boot, [.025, .975]).tolist(),
        "repaired": final["first_errors_repaired"], "first_errors": final["first_errors"],
        "broken": final["originally_correct"]-final["preserved"], "originally_correct": final["originally_correct"],
        "post_eos_masked": "post_eos_positions_removed" in result,
        "post_eos_positions_removed": result.get("post_eos_positions_removed"),
        "elapsed_seconds": result["elapsed_seconds"], "workload_ratios": workload_ratios,
    })
(root/"progress-head-summary.json").write_text(json.dumps(rows, indent=2)+"\n")
lines = [
    "# Native verification-error correction heads", "",
    "These heads modify native draft hidden states before the frozen vocabulary projection. Training uses actual native rollout labels only through the first rejection; the frozen target verifies every proposed token at inference. A differentiable soft-prefix objective is a surrogate, not a measured acceptance probability.", "",
    "Throughput includes full generation wall time, including prefill and correction overhead. Each fit is screened on two requests per GSM8K, MATH, HumanEval and MTBench, a 512-token cap, and two rotated timing repeats. Paired intervals resample requests, clustering timing repeats; they exclude fitting-seed and adaptive-selection uncertainty. The 128-request confirmation remains reserved.", "",
    "| Run/lane | Head | Train questions | Candidate / native TPS | Ratio [95% interval] | Validation errors repaired | Previously correct broken | Seconds |",
    "|---|---|---:|---:|---|---:|---:|---:|",
]
for r in rows:
    lines.append(f"| {r['run']}/{r['lane']} | {r['name']} | {r['counts']['train_questions']} | {r['summary']['candidate']['tps']:.1f} / {r['summary']['native']['tps']:.1f} | {r['native_ratio']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] | {r['repaired']}/{r['first_errors']} | {r['broken']}/{r['originally_correct']} | {r['elapsed_seconds']:.0f} |")
lines += ["", "Protocol note: jobs 29228 and 29235 did not remove post-EOS cached training positions. From job 29271, the EOS label remains supervised but later positions are masked; 159 positions are removed from the mixed cache. Their recorded decoding TPS remains valid, but these fitting protocols are not perfectly matched. Mixed fits also switch to question-uniform sampling and include code and general instructions. Repair/break counts come from different validation pools and must not be compared as absolute rates across pools.", "", "A repaired first rejection can extend a draft, while breaking an earlier accepted token can shorten it. Counts diagnose this tradeoff but do not establish its causal contribution to runtime. The recorded full decoding measurements decide promotion."]
(root/"PROGRESS_HEADS.md").write_text("\n".join(lines)+"\n")
if rows:
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, .43*len(rows))), layout="constrained")
    y = np.arange(len(rows))
    ratios = np.asarray([r["native_ratio"] for r in rows])
    intervals = np.asarray([r["ci95"] for r in rows])
    axes[0].errorbar(ratios, y, xerr=np.maximum(0, np.stack([ratios-intervals[:, 0], intervals[:, 1]-ratios])), fmt="o", capsize=3)
    axes[0].axvline(1., color="gray", linestyle="--", label="Native parity")
    axes[0].axvline(1.1, color="green", linestyle=":", label="Required gain")
    axes[0].set_yticks(y, [r["name"] for r in rows])
    axes[0].set_xlabel("Full generation throughput / native DFlash")
    axes[0].legend(loc="best")
    axes[1].barh(y-.17, [r["repaired"]/max(r["first_errors"], 1)*100 for r in rows], height=.32, label="First errors repaired (%)")
    axes[1].barh(y+.17, [r["broken"]/max(r["originally_correct"], 1)*100 for r in rows], height=.32, label="Correct positions broken (%)")
    axes[1].set_yticks(y, [f"{r['counts']['val_questions']} validation questions" for r in rows])
    axes[1].set_xlabel("Rate within each eligible category (%)")
    axes[1].legend(loc="best")
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
    fig.savefig(root/"progress-heads.png", dpi=180)
    fig.savefig(root/"progress-heads.pdf")
print(json.dumps({"completed_fits": len(rows), "best_screen_ratio": max((r["native_ratio"] for r in rows), default=None)}))
