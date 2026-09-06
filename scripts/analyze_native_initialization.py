"""Audit the matched initialization intervention before interpreting its screen."""

import json
import runpy
from pathlib import Path

from relayspec.cached_fit_evidence import digest

root = Path("reports/autoresearch-20260907")
output = root / "native-initialization-summary.json"
analyze = runpy.run_path("scripts/analyze_compact_interface_data.py")["analyze"]
result = analyze(28532, 42, output)
wave = json.loads(Path("configs/autoresearch/20260907/wave42.json").read_text())
for first, second in [(0, 1), (2, 3)]:
    a, b = [wave["lanes"][i]["trial"] for i in [first, second]]
    assert {k: v for k, v in a.items() if k not in ["name", "initialization"]} == {
        k: v for k, v in b.items() if k not in ["name", "initialization"]
    }
for i, cell in enumerate(result["results"]):
    trial = cell["fit"]["trial"]
    p = (
        root
        / "run-28532"
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
    "Native compact late-layer mapper, matched16/128-record random versus released-column initialization. Seed1729,8192 updates, unchanged teacher/width/normalization and record prefixes. Eight exposed GSM8K questions512token cap, each compared with original512-record compact reference. Untrained cropping and cross-domain controls are reported separately in wave43. No fresh confirmation, minimum-data threshold, or standalone novelty claim."
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
(root / "native-initialization.md").write_text("\n".join(lines) + "\n")
