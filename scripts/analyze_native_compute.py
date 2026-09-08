"""Rebuild exploratory tables and plots from raw native research artifacts."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("reports/native-compute-20260908")
rows = []
for run in sorted(root.glob("run-*")):
    for lane in sorted(run.glob("lane[0-3]")):
        if not (lane / "result.json").exists():
            continue
        result = json.loads((lane / "result.json").read_text())
        provenance = json.loads((lane / "provenance.json").read_text())
        spec = provenance["spec"]
        raw = [json.loads(line) for line in (lane / "records.jsonl").read_text().splitlines()]
        baseline = {r["problem_id"]: r for r in raw if r["method"] == "native"}
        assert result["status"] == "pass"
        assert len(baseline) == spec["requests"]
        assert {r["method"] for r in raw} == {"native", "duplicate", *spec["candidates"]}
        for name in {r["method"] for r in raw}:
            arm = [r for r in raw if r["method"] == name]
            assert len(arm) == spec["requests"]
            assert {r["problem_id"] for r in arm} == set(baseline)
            assert all(r["request_seconds"] > 0 and len(r["output_token_ids"]) == r["output_tokens"] for r in arm)
        for row in raw:
            if row["method"] == "duplicate":
                assert all(row[k] == baseline[row["problem_id"]][k] for k in ["output_token_ids", "acceptance_lengths"])
        base_progress = result["summary"]["methods"]["native"]["progress_per_cycle"]
        for name, point in result["summary"]["methods"].items():
            candidate = spec["candidates"].get(name, {"kind": name})
            rows.append(dict(job=provenance["job"], lane=lane.name, benchmark=spec.get("benchmark", "gsm8k"),
                             requests=spec["requests"], output_cap=spec["max_new_tokens"],
                             method=name, kind=candidate["kind"], candidate=json.dumps(candidate, sort_keys=True),
                             tokens_per_second=point["tokens_per_second"], native_ratio=point["throughput_ratio"],
                             ci_low=point["throughput_ci95"][0], ci_high=point["throughput_ci95"][1],
                             progress_per_cycle=point["progress_per_cycle"], progress_ratio=point["progress_per_cycle"]/base_progress,
                             exact_native_sequences=result["exact_sequences_vs_native"][name],
                             mean_output_tokens=point["mean_output_tokens"], artifact=str(lane)))
with (root / "all-results.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
(root / "all-results.json").write_text(json.dumps(rows, indent=2)+"\n")

palette = {"drop_layers": "#9467bd", "zero_mlp": "#8c564b", "linear_mlp": "#d1495b", "window": "#edae49", "pruned_mlp": "#30638e", "shared_context": "#00798c", "folded_mlp": "#a23b72", "block_size": "#888888", "prefix_refine": "#228833"}
fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.3), layout="constrained")
for ax, benchmark in zip(axes.flat, ["gsm8k", "math500", "humaneval", "mtbench"]):
    points = [r for r in rows if r["requests"] >= 8 and r["benchmark"] == benchmark and r["kind"] in palette]
    for kind, color in palette.items():
        group = [r for r in points if r["kind"] == kind]
        if group:
            ax.scatter([r["progress_ratio"] for r in group], [r["native_ratio"] for r in group], label=kind.replace("_", " "), c=color, s=38, alpha=.8)
    ax.axhline(1, color="black", lw=1, ls="--")
    ax.axvline(1, color="black", lw=1, ls="--")
    ax.plot(1, 1, "k*", ms=12, label="native baseline")
    ax.set(title=benchmark.upper(), xlabel="Accepted progress per cycle / native", ylabel="End-to-end throughput / native")
    ax.grid(alpha=.18); ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Native-compute exploration: reduced compute must preserve accepted progress", fontsize=12)
fig.savefig(root / "native-compute-tradeoffs.png", dpi=200)
fig.savefig(root / "native-compute-tradeoffs.pdf")
plt.close(fig)

lines = ["# Native-compute experiment results", "", "All values are paired against the same released Qwen3-8B native DFlash on one L40S per lane. Four GPUs run independent lanes. These are exposed development requests, one fitting seed, greedy BF16/SDPA, and capped outputs. Request bootstrap intervals exclude seed and selection uncertainty. Exact native agreement is not a guarantee of identity to separately executed AR. Where exact sequences differ, throughput is not a comparison on identical generated work; no answer-quality equivalence is implied. MT-Bench uses only the first user turn.", "", "| Job | Lane | Task | Requests / cap | Candidate | TPS | Native ratio (95% request interval) | Exact native sequences |", "|---|---|---|---|---|---:|---:|---:|"]
for r in rows:
    lines.append(f"| {r['job']} | {r['lane']} | {r['benchmark']} | {r['requests']} / {r['output_cap']} | {r['method']} | {r['tokens_per_second']:.1f} | {r['native_ratio']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {r['exact_native_sequences']}/{r['requests']} |")
lines += ["", "The candidate specification and raw artifact path for every row are in `all-results.csv`. Layer subsets differ between lanes; method names alone are not a complete specification. The plot excludes the two-request smoke screen."]
(root / "RESULTS.md").write_text("\n".join(lines)+"\n")
print(json.dumps({"rows": len(rows), "candidate_evaluations": sum(r["kind"] in palette for r in rows), "jobs": sorted({r["job"] for r in rows})}))
