"""Primary paired comparison for checkpoints recovered with original RoPE state."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("reports")
rows = []
for path in sorted(root.glob("run-*/lane?/recovery-result.json")):
    result = json.loads(path.read_text())
    assert result["status"] == "pass" and result["reload_gate"] == "pass"
    raw = json.loads((path.parent/"evaluation.json").read_text())
    methods = {}
    for name in result["summary"]:
        methods[name] = sorted([r for r in raw if r["method"] == name], key=lambda r: r["problem_id"])
    baseline = methods["native"]
    for name, arm in methods.items():
        assert [r["problem_id"] for r in arm] == [r["problem_id"] for r in baseline]
        if name == "duplicate":
            assert all(all(r[k] == b[k] for k in ["tokens", "acceptance_lengths"]) for r, b in zip(arm, baseline))
    arm = methods["correct_student"]
    indices = np.random.default_rng(1729).integers(0, len(arm), size=(10000, len(arm)))
    def rates(group):
        return np.array([r["output_tokens"] for r in group])[indices].sum(1)/np.array([r["seconds"] for r in group])[indices].sum(1)
    ratio = rates(arm)/rates(baseline)
    rows.append({"artifact": str(path), "config": result["config"], "recovery": result["recovery"], "native_tps": result["summary"]["native"]["tps"], **result["summary"]["correct_student"], "ci95": np.quantile(ratio, [.025, .975]).tolist()})
(root/"corrected-summary.json").write_text(json.dumps(rows, indent=2)+"\n")
text = ["# Corrected native joint-training comparisons", "", "These checkpoint evaluations preserve the original FP32 positional-frequency buffers. Every checkpoint passed fresh-load decoding reproduction and native duplicate controls. They supersede the original after-training performance rows affected by whole-model dtype conversion. All are single-seed, eight-request, 512-token-cap development comparisons; intervals omit seed and selection uncertainty.", "", "| Experiment | Updates | Native TPS | Student TPS | Student/native [95% paired interval] | Exact native |", "|---|---:|---:|---:|---:|---:|"]
for r in rows:
    text.append(f"| {r['config']['name']} | {r['config']['steps']} | {r['native_tps']:.1f} | {r['tps']:.1f} | {r['native_ratio']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] | {r['exact_native']}/{r['requests']} |")
text += ["", "No final superiority/inferiority decision is made from this development subset. Larger data, teacher/objective variations, architecture controls, training seeds and independent confirmation remain in the active goal."]
(root/"CORRECTED_RESULTS.md").write_text("\n".join(text)+"\n")
selected = sorted([r for r in rows if r["config"]["steps"] == 512], key=lambda r: r["config"]["learning_rate"])
fig, ax = plt.subplots(figsize=(6.7, 4), layout="constrained")
x = [r["config"]["learning_rate"] for r in selected]
y = [r["native_ratio"] for r in selected]
ax.errorbar(x, y, yerr=[[r["native_ratio"]-r["ci95"][0] for r in selected], [r["ci95"][1]-r["native_ratio"] for r in selected]], marker="o", capsize=4)
ax.axhline(1, color="black", linestyle="--", label="Original DFlash")
ax.set(xscale="log", xlabel="Learning rate", ylabel="Throughput / original DFlash", title="Two-tap joint training: 128 records, 512 updates")
ax.grid(alpha=.2); ax.legend()
fig.savefig(root/"corrected-rates.png", dpi=200)
print(json.dumps({"recovered_checkpoints": len(rows), "ratios": [round(r["native_ratio"],3) for r in rows]}))
