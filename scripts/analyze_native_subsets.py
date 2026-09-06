"""Verify disjoint training-record windows against the frozen cache index."""

import json
import runpy
from pathlib import Path

from relayspec.cached_fit_evidence import digest

root = Path("reports/autoresearch-20260907")
output = root / "native-subset-summary.json"
analyze = runpy.run_path("scripts/analyze_compact_interface_data.py")["analyze"]
result = analyze(28538, 47, output)
p = root / "native-subset-cache-prefix.json"
cache = json.loads(p.read_text())
assert (
    cache["cache_index_sha256"]
    == "a2fd012df1da5554f8e211cf0b60aec098d3754d866bc50e9623f3117f43f8e5"
)
result["input_sha256"][str(p)] = digest(p)
seen = {e["record_sha256"] for e in cache["entries"][:16]}
for i, cell in enumerate(result["results"]):
    t = cell["fit"]["trial"]
    offset = t["training_record_offset"]
    p = (
        root
        / "run-28538"
        / f"lane{i}"
        / "fitting"
        / t["name"]
        / "training-selection.json"
    )
    selected = json.loads(p.read_text())
    assert selected["offset"] == offset and selected["count"] == 16
    assert selected["entries"] == cache["entries"][offset : offset + 16]
    ids = {e["record_sha256"] for e in selected["entries"]}
    assert len(ids) == 16 and not ids.intersection(seen)
    seen.update(ids)
    result["input_sha256"][str(p)] = digest(p)
result["scope"] = (
    "Four disjoint16-record training windows at offsets16,32,48,64, verified against frozen cache entries and disjoint from original first16. Inherited native-column initialization,128 updates and unchanged validation prefix. Eight reused exposed GSM8K questions512token cap. Reference is original512-record compact map. Record-selection development robustness, not independent evaluation or a population minimum-data guarantee."
)
output.write_text(json.dumps(result, indent=2) + "\n")
