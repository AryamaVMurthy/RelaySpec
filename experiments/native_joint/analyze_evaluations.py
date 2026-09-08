"""Summarize completed repeated evaluations; bootstrap requests, not repeats."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


root = Path("reports")
rows = []
for lane in sorted(root.glob("run-*/lane[0-3]")):
    if not (lane/"evaluation-result.json").exists():
        continue
    result = json.loads((lane/"evaluation-result.json").read_text())
    provenance = json.loads((lane/"provenance.json").read_text())
    spec = provenance["spec"]
    assert result["status"] == "pass" and result["repeats_exact"]
    raw = [json.loads(line) for line in (lane/"evaluation.jsonl").read_text().splitlines()]
    expected_repeats = spec.get("repeats", 1)
    methods = list(result["summary"])
    problems = sorted({r["problem_id"] for r in raw})
    assert len(problems) == spec["requests"]
    assert len(raw) == len(problems)*expected_repeats*len(methods)
    indexed = {(r["method"], r["problem_id"], r["repeat"]): r for r in raw}
    assert len(indexed) == len(raw)
    arrays = {}
    for name in methods:
        arrays[name] = []
        for problem in problems:
            group = [indexed[name, problem, repeat] for repeat in range(expected_repeats)]
            assert all(all(r[k] == group[0][k] for k in ["tokens", "acceptance_lengths"]) for r in group)
            arrays[name].append([sum(r["output_tokens"] for r in group), sum(r["seconds"] for r in group)])
        arrays[name] = np.asarray(arrays[name])
    draws = np.random.default_rng(1729).integers(0, len(problems), size=(10000, len(problems)))
    samples = {name: values[draws].sum(1) for name, values in arrays.items()}
    for name in methods:
        values = arrays[name].sum(0)
        native = arrays["native"].sum(0)
        bootstrap = (samples[name][:, 0]/samples[name][:, 1])/(samples["native"][:, 0]/samples["native"][:, 1])
        mismatches = []
        for problem in problems:
            tokens = indexed[name, problem, 0]["tokens"]
            reference = indexed["native", problem, 0]["tokens"]
            if tokens != reference:
                prefix = next((i for i, (a, b) in enumerate(zip(tokens, reference)) if a != b), min(len(tokens), len(reference)))
                mismatches.append({"problem_id": problem, "first_divergence": prefix, "tokens": len(tokens), "native_tokens": len(reference)})
        rows.append({"run": lane.parent.name, "lane": lane.name, "name": spec["name"], "benchmark": spec["benchmark"], "phase": spec["phase"], "checkpoint_sha256": spec["checkpoint_sha256"], "method": name, "requests": len(problems), "repeats": expected_repeats, "tps": values[0]/values[1], "native_tps": native[0]/native[1], "native_ratio": (values[0]/values[1])/(native[0]/native[1]), "ci95": np.quantile(bootstrap, [.025, .975]).tolist(), "exact_native_requests": len(problems)-len(mismatches), "mismatches": mismatches, "block_size": spec["block_size"] if name == "student" else (1 if name == "ar" else spec.get("native_block_size", 16))})
(root/"evaluation-summary.json").write_text(json.dumps(rows, indent=2)+"\n")
report = ["# Paired standalone evaluation", "", "Intervals resample paired requests with all timing repeats kept together. They exclude fitting-seed and adaptive-selection uncertainty. Throughput includes prefill. Exact outputs compare token arrays with native DFlash; AR differences, if present, are recorded separately and are not a quality score.", "", "| Run/lane | Workload | Method | Block | Requests × repeats | TPS | Native TPS | Ratio [95% interval] | Exact native requests |", "|---|---|---|---:|---:|---:|---:|---|---:|"]
for row in rows:
    if row["method"] == "native":
        continue
    report.append(f"| {row['run']}/{row['lane']} | {row['benchmark']} | {row['method']} | {row['block_size']} | {row['requests']} × {row['repeats']} | {row['tps']:.1f} | {row['native_tps']:.1f} | {row['native_ratio']:.3f} [{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}] | {row['exact_native_requests']}/{row['requests']} |")
(root/"EVALUATIONS.md").write_text("\n".join(report)+"\n")
points = [r for r in rows if r["method"] == "student"]
if points:
    fig, ax = plt.subplots(figsize=(8, max(3.2, len(points)*.38)), layout="constrained")
    for i, point in enumerate(points):
        ax.plot(point["ci95"], [i, i], color="#21639b")
        ax.scatter(point["native_ratio"], i, color="#21639b")
    ax.set_yticks(range(len(points)), [f"{r['run'][4:]}/{r['lane']}: {r['benchmark']} b{r['block_size']}" for r in points])
    ax.axvline(1, color="black", lw=1, ls="--")
    ax.set(xlabel="End-to-end throughput / original DFlash (paired request interval)")
    ax.grid(axis="x", alpha=.2)
    fig.savefig(root/"paired-evaluations.png", dpi=180)
    fig.savefig(root/"paired-evaluations.pdf")
print(json.dumps({"completed_lanes": len({(r['run'], r['lane']) for r in rows}), "method_summaries": len(rows)}))
