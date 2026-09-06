"""Audit matched short random fits and inherited-initialization seed replication."""

import json
import runpy
from pathlib import Path

from relayspec.cached_fit_evidence import digest

root = Path("reports/autoresearch-20260907")
output = root / "native-short-controls-summary.json"
analyze = runpy.run_path("scripts/analyze_compact_interface_data.py")["analyze"]
result = analyze(28536, 45, output)
wave = json.loads(Path("configs/autoresearch/20260907/wave44.json").read_text())
for i, cell in enumerate(result["results"]):
    t = cell["fit"]["trial"]
    ref = next(
        s["trial"]
        for s in wave["lanes"]
        if s["trial"]["distinct_examples"] == t["distinct_examples"]
        and s["trial"]["steps"] == 128
    )
    excluded = {
        "name",
        "study",
        "seed",
        "initialization",
        "native_teacher",
        "native_teacher_sha256",
    }
    assert {k: v for k, v in t.items() if k not in excluded} == {
        k: v for k, v in ref.items() if k not in excluded
    }
    p = root / "run-28536" / f"lane{i}" / "fitting" / t["name"] / "initialization.json"
    record = json.loads(p.read_text())
    assert record["mode"] == t["initialization"]
    if record["mode"] == "native_columns":
        assert record["selected_blocks"] == [3, 4]
        assert record["native_teacher_sha256"] == t["native_teacher_sha256"]
    result["input_sha256"][str(p)] = digest(p)
result["scope"] = (
    "Matched128-update initialization controls at16/128 records and seed1730 inherited-fit repeat execution. Compare wave44 seed1729 inherited fits with unchanged optimizer, record prefixes and teacher. Eight reused exposed GSM8K questions,512token cap and same512-record compact reference. Inherited repeats have identical tensor storage because column copying removes the random initialization and record order is fixed. They do not measure fitting-seed variation or add independent evaluation questions. No fresh confirmation."
)
output.write_text(json.dumps(result, indent=2) + "\n")
