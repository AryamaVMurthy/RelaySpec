"""Audit all predeclared low-rank native calibration arms."""

import json
import runpy
from pathlib import Path

from relayspec.cached_fit_evidence import digest

root = Path("reports/autoresearch-20260907")
output = root / "native-residual-summary.json"
analyze = runpy.run_path("scripts/analyze_compact_interface_data.py")["analyze"]
result = analyze(28545, 52, output)
for i, cell in enumerate(result["results"]):
    trial = cell["fit"]["trial"]
    p = (
        root
        / "run-28545"
        / f"lane{i}"
        / "fitting"
        / trial["name"]
        / "initialization.json"
    )
    init = json.loads(p.read_text())
    assert init["mode"] == "native_columns"
    assert init["selected_blocks"] == [3, 4]
    assert init["native_residual_rank"] == trial["native_residual_rank"]
    assert init["native_teacher_sha256"] == trial["native_teacher_sha256"]
    assert init["residual_scaling"] == 1.0
    cell["initialization"] = init
    result["input_sha256"][str(p)] = digest(p)
result["scope"] = (
    "Four declared residual-rank/rate settings,128 fitting records and128 updates. Frozen native columns plus trainable low-rank correction; export folds into ordinary dense projection. Fewer trainable parameters, not fewer deployed parameters. Eight exposed GSM8K requests512token cap relative to original512-record two-layer reference. No new confirmation or equal-wall-time claim."
)
output.write_text(json.dumps(result, indent=2) + "\n")
