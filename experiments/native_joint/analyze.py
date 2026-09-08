"""Validate paired artifacts and summarize every completed standalone run."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("reports")
rows = []
issues = json.loads((root/"protocol-issue.json").read_text()) if (root/"protocol-issue.json").exists() else {"affected_jobs": []}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
for run in sorted(root.glob("run-*")):
    for lane in sorted(run.glob("lane[0-3]")):
        if not (lane/"result.json").exists():
            continue
        d = json.loads((lane/"result.json").read_text())
        config = json.loads((lane/"provenance.json").read_text())["config"]
        assert d["status"] == "pass"
        for stage in ["before", "after"]:
            raw = json.loads((lane/f"decode-{stage}.json").read_text())
            groups = {name: sorted([r for r in raw if r["method"] == name], key=lambda r: r["problem_id"]) for name in ["native", "duplicate", "student"]}
            baseline = groups["native"]
            assert len(baseline) == config["eval_requests"]
            for name, group in groups.items():
                assert [r["problem_id"] for r in group] == [r["problem_id"] for r in baseline]
                if name == "duplicate":
                    assert all(all(r[k] == b[k] for k in ["tokens", "acceptance_lengths"]) for r, b in zip(group, baseline))
                times = np.array([r["seconds"] for r in group])
                tokens = np.array([r["output_tokens"] for r in group])
                base_times = np.array([r["seconds"] for r in baseline])
                base_tokens = np.array([r["output_tokens"] for r in baseline])
                indices = np.random.default_rng(1729).integers(0, len(group), size=(10000, len(group)))
                ratios = (tokens[indices].sum(1)/times[indices].sum(1))/(base_tokens[indices].sum(1)/base_times[indices].sum(1))
                row = {"run": run.name, "lane": lane.name, "name": config["name"], "config": config, "stage": stage, "method": name, "eligible_for_claim": not (run.name[4:] in issues["affected_jobs"] and stage == "after"), **d[stage][name], "ci95": np.quantile(ratios, [.025, .975]).tolist()}
                rows.append(row)
        # Plot one matched objective/protocol only. CE, target KL, native KL,
        # position weighting and prefix conditioning have incomparable scales.
        if (config["steps"] >= 128 and config.get("teacher") == "native"
                and config.get("loss") == "kl" and config.get("gamma") == 7.
                and not config.get("conditioning_prefix") and config.get("val_records") == 32):
            trajectory = d["validation"]
            label = f"{run.name[4:]}/{lane.name}: {config['name']}"
            axes[0].plot([r["step"] for r in trajectory], [r["loss"] for r in trajectory], marker=".", label=label)
            after = next(r for r in reversed(rows) if r["stage"] == "after" and r["method"] == "student")
            if after["eligible_for_claim"]:
                axes[1].scatter(trajectory[-1]["loss"], after["native_ratio"], label=label)
axes[0].set(xlabel="Optimizer updates", ylabel="Validation native-teacher KL", title="Native KL, gamma=7, 32 validation records")
axes[1].set(xlabel="Final validation native-teacher KL", ylabel="Throughput / original DFlash", title="Same objective; decoding decides performance")
axes[1].axhline(1, color="black", ls="--", lw=1)
for ax in axes:
    ax.grid(alpha=.2)
axes[0].legend(fontsize=6)
fig.savefig(root/"joint-training.png", dpi=180)
fig.savefig(root/"joint-training.pdf")
(root/"summary.json").write_text(json.dumps(rows, indent=2)+"\n")
text = ["# Standalone native joint-training results", "", "**Protocol correction:** original after-training rows from jobs 29153, 29154 and 29155 used rounded nonpersistent RoPE buffers and are excluded from promotion. See `protocol-issue.json`. Corrected checkpoint evaluations are recorded separately as `recovery-result.json`.", "", "Adaptive development screens with one training seed unless explicitly varied. Intervals are paired request bootstrap intervals and exclude training-seed and selection uncertainty. Throughput includes prefill. Output identity is against the pinned native decoder, not a separate AR guarantee. The before/after native controls are remeasured to reduce time-order bias.", "", "| Run/lane | Experiment | Updates | Stage | Student TPS | Native TPS | Student/native [95% interval] | Exact native outputs |", "|---|---|---:|---|---:|---:|---:|---:|"]
for r in rows:
    if r["method"] != "student":
        continue
    native = next(x for x in rows if all(x[k] == r[k] for k in ["run", "lane", "stage"]) and x["method"] == "native")
    text.append(f"| {r['run']}/{r['lane']} | {r['name']} | {r['config']['steps']} | {r['stage']} | {r['tps']:.1f} | {native['tps']:.1f} | {r['native_ratio']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] | {r['exact_native']}/{r['requests']} |")
(root/"RESULTS.md").write_text("\n".join(text)+"\n")
print(json.dumps({"completed_fits": len(rows)//6, "paired_method_summaries": len(rows)}))
