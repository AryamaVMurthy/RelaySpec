"""Audit short native calibration fits against the fixed long-fit recipe."""

import json
import runpy
from pathlib import Path

from relayspec.cached_fit_evidence import digest

root = Path("reports/autoresearch-20260907")
output = root / "native-initialization-budget-summary.json"
analyze = runpy.run_path("scripts/analyze_compact_interface_data.py")["analyze"]
result = analyze(28535, 44, output)
long = json.loads(Path("configs/autoresearch/20260907/wave42.json").read_text())
for i, cell in enumerate(result["results"]):
    trial = cell["fit"]["trial"]
    reference = next(
        s["trial"]
        for s in long["lanes"]
        if s["trial"]["distinct_examples"] == trial["distinct_examples"]
        and s["trial"]["initialization"] == "native_columns"
    )
    exclusions = {
        "name",
        "study",
        "steps",
        "checkpoint_steps",
        "budget_panels",
        "native_teacher",
        "native_teacher_sha256",
    }
    assert {k: v for k, v in trial.items() if k not in exclusions} == {
        k: v for k, v in reference.items() if k not in exclusions
    }
    p = (
        root
        / "run-28535"
        / f"lane{i}"
        / "fitting"
        / trial["name"]
        / "initialization.json"
    )
    init = json.loads(p.read_text())
    assert init["mode"] == "native_columns" and init["selected_blocks"] == [3, 4]
    assert init["native_teacher_sha256"] == trial["native_teacher_sha256"]
    result["input_sha256"][str(p)] = digest(p)
result["scope"] = (
    "Native-column initialized compact maps with16/128 calibration records and128/1024 updates. All non-budget fitting settings match wave42. Eight exposed GSM8K questions512token cap, same512-record compact reference. Development screen, not fresh confirmation or a minimum-data guarantee. Timing excludes original drafter training and cached feature extraction."
)
output.write_text(json.dumps(result, indent=2) + "\n")
