"""Audit the matched initialization intervention before interpreting its screen."""

import json
import runpy
from pathlib import Path

from relayspec.cached_fit_evidence import digest

root = Path("reports/autoresearch-20260907")
output = root / "native-single-tap-summary.json"
analyze = runpy.run_path("scripts/analyze_compact_interface_data.py")["analyze"]
result = analyze(28544, 51, output)
wave = json.loads(Path("configs/autoresearch/20260907/wave51.json").read_text())
for first, second in [(0, 1), (2, 3)]:
    a, b = [wave["lanes"][i]["trial"] for i in [first, second]]
    assert {k: v for k, v in a.items() if k not in ["name", "initialization"]} == {
        k: v for k, v in b.items() if k not in ["name", "initialization"]
    }
for i, cell in enumerate(result["results"]):
    trial = cell["fit"]["trial"]
    p = (
        root
        / "run-28544"
        / f"lane{i}"
        / "fitting"
        / trial["name"]
        / "initialization.json"
    )
    record = json.loads(p.read_text())
    assert record["mode"] == trial["initialization"]
    if record["mode"] == "native_columns":
        assert record["selected_blocks"] == [4]
        assert record["selected_taps"] == [33]
        assert record["native_teacher_sha256"] == trial["native_teacher_sha256"]
    cell["initialization"] = record
    result["input_sha256"][str(p)] = digest(p)
result["scope"] = (
    "Matched128-record single-layer random versus inherited native initialization,128/1024 updates. Eight exposed GSM8K requests512token cap, with original512-record two-layer map as reference. No fresh confirmation."
)
output.write_text(json.dumps(result, indent=2) + "\n")
lines = [
    "# Single-layer native initialization development screen",
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
(root / "native-single-tap.md").write_text("\n".join(lines) + "\n")
