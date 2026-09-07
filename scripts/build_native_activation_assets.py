"""Reproduce all native-compression breadth outcomes from scored records."""

import json
from pathlib import Path

from relayspec.ar_paper_evidence import summarize
from relayspec.cached_fit_evidence import digest

root = Path("reports/autoresearch-20260907")
summary = json.loads((root / "native-activation-breadth-summary.json").read_text())
for p, sha in summary["input_sha256"].items():
    assert digest(Path(p)) == sha
rows = [
    json.loads(s)
    for s in (root / "native-activation-breadth-scored.jsonl").read_text().splitlines()
]
assert len(rows) == 640
lines = [
    r"\begin{tabular}{lllrrr}",
    r"\toprule",
    r"Family & Task & Interface & Tokens/s & Native retained (95\% CI) & Correct \\",
    r"\midrule",
]
for cell in summary["results"]:
    subset = [
        r
        for r in rows
        if r["study_family"] == cell["family"]
        and r["study_workload"] == cell["workload"]
    ]
    rebuilt = summarize(subset, reference=cell["native"])
    assert rebuilt == cell["native_reference"]
    assert summarize(subset, reference="relay_weight1536") == cell["weight_reference"]
    assert len(subset) == 160 and len({r["problem_id"] for r in subset}) == 32
    for method, label in [
        (cell["native"], "Released"),
        ("relay_base", "Repacked full"),
        ("relay_two", "Two layers"),
        ("relay_weight1536", "Weight SVD"),
        ("relay_activation1536", "Activation SVD"),
    ]:
        m = rebuilt["methods"][method]
        lo, hi = m["throughput_ci95"]
        family = "DFlash" if cell["family"] == "dflash" else "EAGLE-3"
        task = "GSM8K" if cell["workload"] == "gsm8k" else "MBPP"
        lines.append(
            f"{family} & {task} & {label} & {m['tokens_per_second']:.2f} & {100 * m['throughput_ratio']:.2f} [{100 * lo:.2f},{100 * hi:.2f}]\\% & {round(32 * m['accuracy'])}/32 "
            + r"\\"
        )
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines.append(r"\end{tabular}")
Path("paper/iclr2027/generated/native_activation_table.tex").write_text(
    "\n".join(lines) + "\n"
)
