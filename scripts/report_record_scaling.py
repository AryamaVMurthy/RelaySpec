"""Plot completed, scored distinct-record scaling cells without mixing datasets."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/data-scaling-20260906"


def main():
    ledger = json.loads((BASE / "jobs.json").read_text())
    fit = next(j for j in ledger["jobs"] if j["kind"] == "cached_fit")
    full = next(j for j in ledger["jobs"] if j["kind"] == "data_evaluation_full")
    fit_path, eval_path = Path(fit["local"]), Path(full["local"])
    gate = json.loads((fit_path / "batch-gate.json").read_text())
    if gate["status"] != "pass":
        raise ValueError("Fitting gate failed")
    analysis = json.loads((eval_path / "analysis.json").read_text())
    methods = analysis["tasks"]["math500"]["methods"]
    fixed = [(n, f"relay_n{n}_fixed") for n in [512, 2048, 4096, 8192, 16384, 32768]]
    epochs = [
        (n, f"relay_n{n}_epoch1" if n != 32768 else "relay_n32768_fixed")
        for n in [4096, 8192, 16384, 32768]
    ]
    panels = {"fixed_8192_updates": fixed, "one_epoch": epochs}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    summary = {}
    for ax, (label, points) in zip(axes, panels.items(), strict=True):
        xs = [n for n, _ in points]
        ys = [methods[m]["throughput_ratio"] for _, m in points]
        ci = [methods[m]["throughput_ci95"] for _, m in points]
        ax.errorbar(
            xs,
            ys,
            yerr=[
                [y - c[0] for y, c in zip(ys, ci)],
                [c[1] - y for y, c in zip(ys, ci)],
            ],
            marker="o",
            capsize=4,
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(n) for n in xs], rotation=40)
        ax.set_xlabel("Distinct training records")
        ax.set_ylabel("End-to-end throughput / AR")
        ax.set_title(
            "8,192 updates per fit"
            if label.startswith("fixed")
            else "One epoch per dataset"
        )
        ax.grid(alpha=0.25)
        summary[label] = {str(n): methods[m] for n, m in points}
    fig.suptitle("NuminaMath training; 128 development questions; one fitting seed")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(BASE / f"distinct-record-scaling.{ext}", dpi=200)
    plt.close(fig)
    diagnostics = {}
    for folder in sorted((fit_path / "fitting").iterdir()):
        complete = json.loads((folder / "fit-complete.json").read_text())
        trial = complete["trial"]
        if complete["distinct_records_seen"] != trial["distinct_examples"]:
            raise ValueError("Claimed training count exceeds actually seen records")
        diagnostics[str(trial["distinct_examples"])] = [
            json.loads(line)
            for line in (folder / "validation.jsonl").read_text().splitlines()
        ]
    sources = [
        BASE / "jobs.json",
        fit_path / "batch-gate.json",
        eval_path / "analysis.json",
    ]
    result = {
        "status": "complete",
        "panels": summary,
        "fitting_diagnostics": diagnostics,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "scope": "Nested NuminaMath subsets, single seed. Request bootstrap intervals exclude fitting-seed variation. Separate from historical MATH-only scaling.",
    }
    (BASE / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
