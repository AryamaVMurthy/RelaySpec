"""Audit two-scalar native calibration and its folded decoding checkpoints."""

import json
import runpy
from pathlib import Path

from relayspec.cached_fit_evidence import digest

root = Path("reports/autoresearch-20260907")
output = root / "native-block-gains-summary.json"
analyze = runpy.run_path("scripts/analyze_compact_interface_data.py")["analyze"]
result = analyze(28539, 48, output)
for i, cell in enumerate(result["results"]):
    t = cell["fit"]["trial"]
    folder = root / "run-28539" / f"lane{i}" / "fitting" / t["name"]
    p = folder / "block-gains.json"
    gain = json.loads(p.read_text())
    complete = json.loads((folder / "fit-complete.json").read_text())
    assert gain["trainable_parameters"] == 2 and complete["trainable_parameters"] == 2
    assert t["native_block_gains"] and t["initialization"] == "native_columns"
    assert gain["steps"] == 128 and len(gain["gains"]) == 2
    cell["block_gains"] = gain
    result["input_sha256"][str(p)] = digest(p)
result["scope"] = (
    "Two trainable layer gains on frozen selected native projection columns. Gains folded into standard33.55M-weight dense deployment, with no added inference operation.16/128 calibration records,128 updates, two scalar learning rates0.003/0.03. Eight exposed GSM8K requests512token cap, original512-record compact reference. No matched-rate/full-budget superiority, fresh confirmation or cross-domain claim."
)
output.write_text(json.dumps(result, indent=2) + "\n")
