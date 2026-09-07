"""Audit the matched initialization intervention before interpreting its screen."""

import json
import runpy
from pathlib import Path

from relayspec.cached_fit_evidence import digest

root = Path("reports/autoresearch-20260907")
output = root / "native-eagle-budget-summary.json"
analyze = runpy.run_path("scripts/analyze_compact_interface_data.py")["analyze"]
result = analyze(
    28556,
    61,
    output,
    cache_sha="9c18d186420ff569186fc1232b411cbb984ae05bbe09564a36f6179ceaccad19",
)
wave = json.loads(Path("configs/autoresearch/20260907/wave61.json").read_text())
for first, second in [(0, 1), (2, 3)]:
    a, b = [wave["lanes"][i]["trial"] for i in [first, second]]
    assert {k: v for k, v in a.items() if k not in ["name", "initialization"]} == {
        k: v for k, v in b.items() if k not in ["name", "initialization"]
    }
for i, cell in enumerate(result["results"]):
    trial = cell["fit"]["trial"]
    p = (
        root
        / "run-28556"
        / f"lane{i}"
        / "fitting"
        / trial["name"]
        / "initialization.json"
    )
    record = json.loads(p.read_text())
    assert record["mode"] == trial["initialization"]
    if record["mode"] == "native_columns":
        assert record["selected_blocks"] == [3, 4]
        assert record["selected_taps"] == [25, 33]
        assert record["native_teacher_sha256"] == trial["native_teacher_sha256"]
    cell["initialization"] = record
    result["input_sha256"][str(p)] = digest(p)
result["scope"] = (
    "Native EAGLE3 late-two-layer map with512 records and128/8192 updates. Matched random versus inherited native columns, raw post-fusion interface without external norm. Eight exposed GSM8K requests512token cap, paired full repacked native reference. No cross-family absolute feature-error comparison, no fresh confirmation."
)
output.write_text(json.dumps(result, indent=2) + "\n")
