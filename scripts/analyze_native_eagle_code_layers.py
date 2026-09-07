"""Keep all earlier-layer additions and compare paired code drafting controls."""

import json
import runpy
from pathlib import Path

from relayspec.ar_paper_evidence import summarize

root = Path("reports/autoresearch-20260907")
output = root / "native-eagle-code-layers-summary.json"
analyze = runpy.run_path("scripts/analyze_compact_interface_data.py")["analyze"]
result = analyze(
    28569,
    74,
    output,
    cache_sha="9c18d186420ff569186fc1232b411cbb984ae05bbe09564a36f6179ceaccad19",
)
for i, cell in enumerate(result["results"]):
    rows = [
        json.loads(s)
        for s in (root / f"run-28569/lane{i}/benchmark-rank0.jsonl")
        .read_text()
        .splitlines()
    ]
    cell["two_layer_reference"] = summarize(rows, reference="relay_two_control")
    cell["cap_counts"] = {
        m: sum(r["output_tokens"] == 256 for r in rows if r["method"] == m)
        for m in cell["decoding"]["methods"]
    }
result["scope"] = (
    "Four predeclared layer sets,512 fitting records128 updates, inherited columns. Eight exposed code requests at256token cap, full native and fixed two-layer controls paired within every worker. No functional correctness or fresh confirmation claim. All layer choices retained, paired intervals exclude fitting and selection uncertainty."
)
output.write_text(json.dumps(result, indent=2) + "\n")
for cell in result["results"]:
    print(
        cell["fit"]["trial"]["target_layer_ids"],
        cell["two_layer_reference"]["methods"]["relay_reduced"],
        cell["cap_counts"],
    )
