"""Audit the matched initialization intervention before interpreting its screen."""

import json
import runpy
from pathlib import Path

from relayspec.cached_fit_evidence import digest

root = Path("reports/autoresearch-20260907")
output = root / "native-eagle-initialization-summary.json"
analyze = runpy.run_path("scripts/analyze_compact_interface_data.py")["analyze"]
result = analyze(
    28552,
    57,
    output,
    cache_sha="9c18d186420ff569186fc1232b411cbb984ae05bbe09564a36f6179ceaccad19",
)
wave = json.loads(Path("configs/autoresearch/20260907/wave57.json").read_text())
for first, second in [(0, 1), (2, 3)]:
    a, b = [wave["lanes"][i]["trial"] for i in [first, second]]
    assert {k: v for k, v in a.items() if k not in ["name", "initialization"]} == {
        k: v for k, v in b.items() if k not in ["name", "initialization"]
    }
for i, cell in enumerate(result["results"]):
    trial = cell["fit"]["trial"]
    p = (
        root
        / "run-28552"
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
    "Native EAGLE3 late-two-layer map with16/128 records and128 updates. Matched random versus inherited native columns, raw post-fusion interface without external norm. Eight exposed GSM8K requests512token cap, paired full repacked native reference. No cross-family absolute feature-error comparison, no fresh confirmation."
)
output.write_text(json.dumps(result, indent=2) + "\n")
lines = [
    "# Native initialization development screen",
    "",
    "| Records | Initialization | Train error | Validation error | Throughput /512-record compact reference |",
    "|---:|---|---:|---:|---:|",
]
for r in result["results"]:
    m = r["decoding"]["methods"]["relay_reduced"]
    lo, hi = m["throughput_ci95"]
    lines.append(
        f"|{r['records']}|{r['initialization']['mode']}|{r['fit']['train_objective']:.6f}|{r['fit']['validation_objective']:.6f}|{m['throughput_ratio']:.4f} [{lo:.4f}, {hi:.4f}]|"
    )
lines += ["", result["scope"]]
(root / "native-eagle-initialization.md").write_text("\n".join(lines) + "\n")
