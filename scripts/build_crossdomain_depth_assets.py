"""Expose cross-domain limits of compact target-layer interfaces."""

import hashlib
import json
from pathlib import Path

root = Path("reports/autoresearch-20260907/run-28427")
rows = []
inputs = {}
for lane, family, task in [
    (0, "DFlash", "Code"),
    (1, "DFlash", "Dialogue"),
    (2, "EAGLE-3", "Code"),
    (3, "EAGLE-3", "Dialogue"),
]:
    path = root / f"lane{lane}" / "research-result.json"
    d = json.loads(path.read_text())
    if d["status"] != "pass" or d["summary"]["clusters"] != 8:
        raise ValueError("Cross-domain lane incomplete")
    inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    methods = d["summary"]["methods"]
    rows.append(
        dict(
            family=family,
            task=task,
            clusters=8,
            requests=d["summary"]["requests"],
            methods={
                k: methods[k] for k in ["relay_early", "relay_spaced", "relay_last2"]
            },
        )
    )
lines = [
    r"\begin{tabular}{llrrr}",
    r"\toprule",
    r"Drafter & Workload & Early & Spaced & Late \\",
    r"\midrule",
]
for r in rows:
    values = " & ".join(
        f"{100 * r['methods'][m]['throughput_ratio']:.1f}\\%"
        for m in ["relay_early", "relay_spaced", "relay_last2"]
    )
    lines.append(f"{r['family']} & {r['task']} & {values} " + r"\\")
lines += [r"\bottomrule", r"\end{tabular}"]
Path("paper/iclr2027/generated/crossdomain_depth_table.tex").write_text(
    "\n".join(lines) + "\n"
)
Path("reports/autoresearch-20260907/crossdomain-depth-summary.json").write_text(
    json.dumps(
        dict(
            inputs=inputs,
            rows=rows,
            scope="Eight exposed MBPP requests,512-token cap, or eight two-turn MT-Bench conversations,256tokens/turn. All maps fit the same512 Numina records. Conditional throughput only, not code execution or dialogue-quality scores. Intervals in JSON resample whole conversations.",
        ),
        indent=2,
    )
    + "\n"
)
