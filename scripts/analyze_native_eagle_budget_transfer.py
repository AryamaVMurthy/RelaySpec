"""Audit native EAGLE low-data initialization with cropping and SVD controls."""

import json
import runpy
from pathlib import Path

root = Path("reports/autoresearch-20260907")
output = root / "native-eagle-budget-transfer-summary.json"
analyze = runpy.run_path("scripts/analyze_native_initialization_transfer.py")["analyze"]
result = analyze(28558, 63, output)
result["scope"] = (
    "Native EAGLE3: eight exposed requests per workload, two turns per dialogue,256token cap. All four512-record random/inherited128/8192-update fits are paired with untrained cropping, full repacked native and rank1536 SVD. Full native export equivalence passed four operational pilots. No new confirmation or full-answer/code/dialogue quality claim. No direct cross-family absolute throughput comparison."
)
output.write_text(json.dumps(result, indent=2) + "\n")
