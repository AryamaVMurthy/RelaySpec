"""Regenerate sampled-decoding table from all frozen, scored question clusters."""

import hashlib
import json
from pathlib import Path

from analyze_sampling_breadth import clustered_summary

root = Path("reports/autoresearch-20260907")
cells = []
for stem, expected_count, family in [
    ("dflash-sampling-matched", 384, "dflash"),
    ("sampling-breadth", 768, "eagle3"),
]:
    summary = json.loads((root / f"{stem}-summary.json").read_text())
    assert summary["status"] == "complete"
    for name, sha in summary["input_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha
    rows = [
        json.loads(s) for s in (root / f"{stem}-scored.jsonl").read_text().splitlines()
    ]
    assert len(rows) == expected_count
    for cell in summary["results"]:
        selected = [
            r
            for r in rows
            if r["study_family"] == cell["family"]
            and r["study_temperature"] == cell["temperature"]
        ]
        rebuilt = clustered_summary(selected, "native_ar", [1730, 1731])
        assert rebuilt == cell["ar_reference"]
        if cell["family"] == family:
            cells.append(cell)
assert len(cells) == 4
lines = [
    r"\begin{tabular}{llrrrrr}",
    r"\toprule",
    r"Family / $T$ & Method & Tokens/s & AR ratio (95\% CI) & Time ratio & Tokens & Correct \\",
    r"\midrule",
]
for cell in cells:
    rebuilt = cell["ar_reference"]
    family = "DFlash" if cell["family"] == "dflash" else "EAGLE-3"
    for method, label in [
        ("native_ar", "AR"),
        (
            "native_target_dflash"
            if cell["family"] == "dflash"
            else "native_target_eagle3",
            "Native",
        ),
        ("relay_base", "RelaySpec"),
    ]:
        m = rebuilt["methods"][method]
        lo, hi = m["throughput_ci95"]
        lines.append(
            f"{family} / {cell['temperature']} & {label} & {m['tokens_per_second']:.2f} & {m['throughput_ratio']:.2f} [{lo:.2f},{hi:.2f}] & {m['request_time_ratio']:.2f} & {m['mean_output_tokens']:.1f} & {m['correct']}/64 "
            + r"\\"
        )
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines.append(r"\end{tabular}")
Path("paper/iclr2027/generated/sampling_breadth_table.tex").write_text(
    "\n".join(lines) + "\n"
)
