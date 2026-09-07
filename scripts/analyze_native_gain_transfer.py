"""Compare every scalar calibration arm directly with cropping and full fitting."""

import json
import runpy
from pathlib import Path

from relayspec.ar_paper_evidence import summarize

root = Path("reports/autoresearch-20260907")
output = root / "native-gain-transfer-summary.json"
analyze = runpy.run_path("scripts/analyze_native_initialization_transfer.py")["analyze"]
result = analyze(28540, 49, output)
for i, cell in enumerate(result["results"]):
    rows = [
        json.loads(s)
        for s in (root / "run-28540" / f"lane{i}" / "benchmark-rank0.jsonl")
        .read_text()
        .splitlines()
    ]
    cell["full_fit_references"] = {
        str(n): summarize(rows, reference=f"relay_n{n}_native_columns")
        for n in [16, 128]
    }
result["scope"] = (
    "Nine methods per worker: full repacked native, untrained cropped native, original512-record compact reference,16/128-record full inherited fits and all four two-scalar fits. Scalar arms use128 updates at rates0.003/0.03, full fits128 updates at0.0006. Eight exposed requests per workload512token cap, with two turns per dialogue conversation. Paired intervals cluster conversations and do not adjust for development selection or multiple comparisons. No new quality confirmation."
)
output.write_text(json.dumps(result, indent=2) + "\n")
lines = [
    "# Two-scalar versus full native calibration",
    "",
    "| Task | Records | Scalar LR | Scalar / crop | Scalar / full fit at same records |",
    "|---|---:|---:|---:|---:|",
]
for cell in result["results"]:
    for lane, (n, lr) in enumerate(
        [(16, 0.003), (16, 0.03), (128, 0.003), (128, 0.03)]
    ):
        name = f"relay_gains_lane{lane}"
        values = []
        for ref in [cell["cropped_reference"], cell["full_fit_references"][str(n)]]:
            m = ref["methods"][name]
            lo, hi = m["throughput_ci95"]
            values.append(f"{m['throughput_ratio']:.4f} [{lo:.4f}, {hi:.4f}]")
        lines.append(f"|{cell['task']}|{n}|{lr}|" + "|".join(values) + "|")
lines += ["", result["scope"]]
(root / "native-gain-transfer.md").write_text("\n".join(lines) + "\n")
