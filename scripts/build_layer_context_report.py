"""Make a readable report from audited layer/context experiments."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("reports/autoresearch-20260907")
short = json.loads((root / "layer-context-wave97-summary.json").read_text())
long = json.loads((root / "layer-context-wave99-summary.json").read_text())
names = {
    "layer_context": "Layer + context",
    "layer_only": "Layer only",
    "context_only": "Context only, five maps",
    "dense_context": "Context only, dense",
}
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for c in long["cells"]:
    f = c["fit"]
    mode = f["settings"]["mode"]
    history = f["history"]
    for ax, key in zip(axes, ["layer", "context"]):
        if key == "layer" and mode == "dense_context":
            continue
        ax.plot(
            [h["step"] for h in history if h["step"] > 0],
            [h["validation"][key] for h in history if h["step"] > 0],
            marker="o",
            label=names[mode],
        )
for ax, key in zip(axes, ["layer", "context"]):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Training updates")
    ax.set_ylabel(f"Validation {key} relative error")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
fig.suptitle("512 records, 32 fixed positions per record, one fitting seed")
fig.tight_layout()
fig.savefig(root / "layer-context-validation.png", dpi=180)
fig.savefig(root / "layer-context-validation.pdf")
plt.close(fig)
breadth = json.loads((root / "layer-context-wave100-summary.json").read_text())
lines = [
    "# User-proposed layer + context loss: controlled test",
    "",
    "DFlash4B drafter retargeted to Qwen3-8B. Four arms use512 training and128 validation records,32 fixed sampled positions per record,2048-position batches,AdamW lr0.0006 with zero decay,seed1729. Five bias-free4096→2560 maps and fixed source fusion/RMSNorm implement the requested joint objective with equal weights and denominator epsilon1e-6. The dense control uses the existing normalized dense architecture. Every arm has52,428,800 trainable weights. Frozen cached sequences are capped at192 tokens; this does not reproduce a512/4k rollout-length study.",
    "",
    "All four arms share identical record/token-position selection. Source taps are regenerated from the cached token IDs. Eight reconstructed teacher checks have relative error below0.00002. Folding five maps into the deployment projection is algebraically checked inFP32; BF16 folding need not be bitwise identical to a two-stage computation.",
    "",
    "## Short screens",
    "",
    "| Arm |128-update retention |1024-update retention |",
    "|---|---:|---:|",
]
for s, longer in zip(short["cells"], long["cells"]):
    mode = longer["fit"]["settings"]["mode"]
    lines.append(
        f"|{names[mode]}|{100 * s['throughput']['methods']['relay_layer_context']['throughput_ratio']:.2f}%|{100 * longer['throughput']['methods']['relay_layer_context']['throughput_ratio']:.2f}%|"
    )
lines += [
    "",
    "Both eight-question screens use the same exposed GSM8K questions. Retention is relative to each worker’s original8192-update RelaySpec reference. All outputs score8/8, no caps. Cross-worker candidate rankings are exploratory.",
    "",
    "## Paired32-question endpoint comparison",
    "",
    "| Arm |Tokens/s |Reference retained [95% CI] |Correct |At cap |",
    "|---|---:|---|---:|---:|",
]
for m, v in breadth["pooled"]["methods"].items():
    rows = [r for c in breadth["cells"] for r in c["rows"] if r["method"] == m]
    lo, hi = v["throughput_ci95"]
    name = (
        "Existing RelaySpec8192-update reference"
        if m == "relay_base"
        else names[m.removeprefix("relay_")]
    )
    lines.append(
        f"|{name}|{v['tokens_per_second']:.2f}|{100 * v['throughput_ratio']:.2f}% [{100 * lo:.2f},{100 * hi:.2f}]|{sum(bool(r['correct']) for r in rows)}/32|{sum(r['output_tokens'] >= 2048 for r in rows)}|"
    )
lines += [
    "",
    "All four1024-update candidates and the original reference run within each worker on identical questions.32 exposed development requests, four8-question shards,2048token cap. Earlier8 are a subset, not an additional independent sample. Confidence intervals resample paired requests and exclude fitting-seed and selection uncertainty. No quality noninferiority claim.",
    "",
    "The equal-weight joint loss is13.0% slower than the existing reference,6.0% slower than the matched-budget dense control, and7.3% slower than context-only with the same five-map structure. Joint/dense ratio0.9400[0.9230,0.9586], joint/context-only0.9271[0.9095,0.9457]. Every method produces identical tokens and scores30/32, with no caps. This experiment does not justify replacing the baseline.",
    "",
    "![Validation trajectories](layer-context-validation.png)",
    "",
]
(root / "LAYER_CONTEXT_LOSS.md").write_text("\n".join(lines))
