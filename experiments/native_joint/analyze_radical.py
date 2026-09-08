"""Screen radical variants, including failures and per-workload regressions."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("reports")
rows, failures, preparation_failures = [], [], []
benchmarks = ["gsm8k", "math500", "humaneval", "mtbench"]
for status_path in sorted(root.glob("run-*/lane[0-3]-status.json")):
    status = json.loads(status_path.read_text())
    lane = status_path.name.split("-")[0]
    config_path = status_path.parent/f"{lane}.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    if status["exit_code"] and config.get("variants") and not list((status_path.parent/lane).glob("*/result.json")):
        preparation_failures.append({"run": status_path.parent.name, "lane": lane, "status": status, "config": config, "log": str(status_path.parent/f"{lane}.log")})
for path in sorted(root.glob("run-*/lane[0-3]/*/result.json")):
    lane_config=path.parents[2]/(path.parents[1].name+'.json')
    if lane_config.exists() and json.loads(lane_config.read_text()).get('phase')=='confirmation':
        continue
    result = json.loads(path.read_text())
    if "variant" not in result:
        continue
    if result["status"] != "pass":
        failures.append({"path": str(path), **result})
        continue
    raw = [json.loads(line) for line in (path.parent/"evaluation.jsonl").read_text().splitlines()]
    for benchmark in ["all"]+benchmarks:
        selected = [r for r in raw if benchmark == "all" or r["benchmark"] == benchmark]
        if not selected:
            continue
        groups = {name: sorted([r for r in selected if r["method"] == name], key=lambda r: r["problem_id"]) for name in ["native", "candidate"]}
        native, candidate = groups["native"], groups["candidate"]
        assert [r["problem_id"] for r in native] == [r["problem_id"] for r in candidate]
        problems = sorted({r["problem_id"] for r in native})
        base, arm = [np.asarray([[sum(r["output_tokens"] for r in group if r["problem_id"] == problem), sum(r["seconds"] for r in group if r["problem_id"] == problem)] for problem in problems]) for group in [native, candidate]]
        draws = np.random.default_rng(1729).integers(0, len(problems), size=(10000, len(problems)))
        a, b = arm[draws].sum(1), base[draws].sum(1)
        ratios = (a[:, 0]/a[:, 1])/(b[:, 0]/b[:, 1])
        trace = [t for r in candidate for t in r["trace"]]
        rows.append({"run": path.parents[2].name, "lane": path.parents[1].name, "variant": result["variant"], "benchmark": benchmark, "requests": len(problems), "timed_generations": len(candidate), "tps": arm[:, 0].sum()/arm[:, 1].sum(), "native_tps": base[:, 0].sum()/base[:, 1].sum(), "native_ratio": (arm[:, 0].sum()/arm[:, 1].sum())/(base[:, 0].sum()/base[:, 1].sum()), "ci95": np.quantile(ratios, [.025, .975]).tolist(), "exact_native": sum(r["tokens"] == n["tokens"] for r, n in zip(candidate, native) if r.get("repeat", 0) == 0), "draft_skip_fraction": sum(not t["draft_call"] for t in trace)/len(trace), "mean_verification_tokens": sum(t["verify_tokens"]*t["branches"] for t in trace)/len(trace), "mean_progress": sum(t["progress"] for t in trace)/len(trace), "elapsed_seconds": result["elapsed_seconds"]})
(root/"radical-summary.json").write_text(json.dumps({"rows": rows, "failures": failures, "preparation_failures": preparation_failures}, indent=2)+"\n")
ranked = sorted([r for r in rows if r["benchmark"] == "all"], key=lambda r: r["native_ratio"], reverse=True)
report = ["# Rapid decoding hypothesis screens", "", "Target: at least10% end-to-end throughput improvement over original DFlash. These adaptive short screens do not establish a win: larger paired evaluation and output-quality checks are required. Intervals are paired request bootstraps and do not include search selection uncertainty. Numerical output identity with original block16 decoding is reported explicitly.", "", "| Run/lane | Hypothesis | Candidate TPS | Original TPS | Ratio [95% interval] | Exact outputs | Draft calls skipped | Seconds for test |", "|---|---|---:|---:|---|---:|---:|---:|"]
for r in ranked:
    report.append(f"| {r['run']}/{r['lane']} | {r['variant']['name']} | {r['tps']:.1f} | {r['native_tps']:.1f} | {r['native_ratio']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] | {r['exact_native']}/{r['requests']} | {r['draft_skip_fraction']:.1%} | {r['elapsed_seconds']:.1f} |")
report += ["", "Selection decision: the run29184 two-code-request history-lookup pilot showed1.332×, but its run29192 follow-up on eight code requests with two timing repeats obtained1.000× [0.947,1.058]. That pilot gain did not replicate and is not a promotion candidate. Ranking below/above by an adaptive point estimate does not establish superiority.", "", f"Failed hypotheses: {len(failures)}. Error traces are retained in radical-summary.json and the raw run folders."]
report += ["", f"Lanes failing before hypothesis timing: {len(preparation_failures)}. These are preparation/engineering failures, not measured regressions."]
for failure in preparation_failures:
    report.append(f"- {failure['run']}/{failure['lane']}: exit{failure['status']['exit_code']}; log `{failure['log']}`.")
(root/"RADICAL_RESULTS.md").write_text("\n".join(report)+"\n")
if ranked:
    fig, ax = plt.subplots(figsize=(8, max(4, len(ranked)*.28)), layout="constrained")
    matrix = np.array([[next((r["native_ratio"] for r in rows if r["run"] == p["run"] and r["lane"] == p["lane"] and r["variant"]["name"] == p["variant"]["name"] and r["benchmark"] == benchmark), float("nan")) for benchmark in benchmarks] for p in ranked])
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=.7, vmax=1.3)
    ax.set_xticks(range(len(benchmarks)), benchmarks)
    ax.set_yticks(range(len(ranked)), [r["variant"]["name"] for r in ranked])
    for i in range(len(ranked)):
        for j in range(len(benchmarks)):
            ax.text(j, i, f"{matrix[i,j]:.2f}×", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Throughput / original DFlash")
    fig.savefig(root/"radical-workloads.png", dpi=180)
    fig.savefig(root/"radical-workloads.pdf")
print(json.dumps({"completed_hypotheses": len(ranked), "failures": len(failures), "best": ranked[:3]}))
