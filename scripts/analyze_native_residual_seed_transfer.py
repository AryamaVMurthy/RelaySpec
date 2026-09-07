"""Compare every low-rank residual arm with untrained and full-fit controls."""

import json
import runpy
from pathlib import Path

from relayspec.ar_paper_evidence import summarize

root = Path("reports/autoresearch-20260907")
output = root / "native-residual-seed-transfer-summary.json"
analyze = runpy.run_path("scripts/analyze_native_initialization_transfer.py")["analyze"]
result = analyze(28553, 58, output)
lines = [
    "# Low-rank native calibration transfer",
    "",
    "|Task|Rank|Seed|Residual / crop (95% CI)|Residual / full fit (95% CI)|",
    "|---|---:|---:|---:|---:|",
]
for i, cell in enumerate(result["results"]):
    rows = [
        json.loads(s)
        for s in (root / "run-28553" / f"lane{i}" / "benchmark-rank0.jsonl")
        .read_text()
        .splitlines()
    ]
    cell["full_fit_reference"] = summarize(rows, reference="relay_n128_native_columns")
    for j, (rank, lr) in enumerate(
        [(16, 1730), (128, 1730), (16, 1731), (128, 1731)]
    ):
        fields = []
        for summary in [cell["cropped_reference"], cell["full_fit_reference"]]:
            m = summary["methods"][f"relay_residual_lane{j}"]
            lo, hi = m["throughput_ci95"]
            fields.append(f"{m['throughput_ratio']:.4f} [{lo:.4f},{hi:.4f}]")
        lines.append(f"|{cell['task']}|{rank}|{lr}|" + "|".join(fields) + "|")
result["scope"] = (
    "All four rank16/128 and seed1730/1731 settings at lr0.006 with128 calibration records and128 updates, compared in each worker to full-matrix fitting at128 records/128 updates/lr0.0006, untrained crop, original512-record compact and repacked native. Eight exposed requests per workload512token cap, with dialogue turns clustered by conversation. No new confirmation, uncapped quality, equal training time or deployed-size reduction claim. No multiple-comparison adjustment."
)
output.write_text(json.dumps(result, indent=2) + "\n")
(root / "native-residual-seed-transfer.md").write_text(
    "\n".join(lines) + "\n\n" + result["scope"] + "\n"
)
