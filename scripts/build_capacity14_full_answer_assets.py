"""Rebuild full-answer14B tables from audited scored request records."""

import json
from pathlib import Path

from relayspec.ar_paper_evidence import summarize
from relayspec.cached_fit_evidence import digest

root = Path("reports/autoresearch-20260907")
d = json.loads((root / "capacity14-full-answer-summary.json").read_text())
assert d["status"] == "complete"
for path, sha in d["input_sha256"].items():
    assert digest(Path(path)) == sha
rows = [
    json.loads(s)
    for s in (root / "capacity14-full-answer-scored.jsonl").read_text().splitlines()
]
assert len(rows) == 320
lines = [
    r"\begin{tabular}{llrrrrr}",
    r"\toprule",
    r"Family & Method & Weights & Tokens/s & Dense retained (95\% CI) & Correct & Cap \\",
    r"\midrule",
]
for c in d["results"]:
    selected = [r for r in rows if r["study_family"] == c["family"]]
    rebuilt = summarize(selected, reference="relay_base")
    assert rebuilt == c["dense2048_reference"]
    family = "DFlash" if c["family"] == "dflash" else "EAGLE-3"
    for method, label, params in [
        ("native_ar", "AR", "--"),
        ("relay_dense_n512", "Dense,512", "65.54M"),
        ("relay_base", "Dense,2048", "65.54M"),
        ("relay_factorized4096_n2048", "Linear4096,2048", "115.34M"),
        ("relay_mlp4096_n2048", "MLP4096,2048", "115.34M"),
    ]:
        m = rebuilt["methods"][method]
        lo, hi = m["throughput_ci95"]
        correct = round(m["accuracy"] * 32)
        lines.append(
            f"{family} & {label} & {params} & {m['tokens_per_second']:.2f} & {100 * m['throughput_ratio']:.2f} [{100 * lo:.2f},{100 * hi:.2f}]\\% & {correct}/32 & {c['cap_counts'][method]} "
            + r"\\"
        )
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines.append(r"\end{tabular}")
Path("paper/iclr2027/generated/capacity14_full_answer_table.tex").write_text(
    "\n".join(lines) + "\n"
)
