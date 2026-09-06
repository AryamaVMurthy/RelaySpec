"""Compare matched-rank compression objectives without conflating adaptation budgets."""

import hashlib
import json
from pathlib import Path

root = Path("reports/autoresearch-20260907")
results = []
inputs = {}
for job, lane, family, task, direct in [
    (28428, 0, "EAGLE-3 8B", "GSM8K", "relay_reduced"),
    (28428, 1, "EAGLE-3 8B", "MATH", "relay_trained1024"),
    (28425, 2, "DFlash 14B", "MATH", "relay_trained1024"),
]:
    path = root / f"run-{job}" / f"lane{lane}" / "research-result.json"
    d = json.loads(path.read_text())
    if d["status"] != "pass" or d["summary"]["requests"] != 32:
        raise ValueError("Incomplete matched-rank screen")
    inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    results.append(dict(family=family, task=task, direct=direct, summary=d["summary"]))
lines = [
    r"\begin{tabular}{llrrr}",
    r"\toprule",
    r"Drafter / verifier & Task & Weight SVD & Output-subspace & Direct fitting \\",
    r"\midrule",
]
for r in results:
    m = r["summary"]["methods"]
    values = " & ".join(
        f"{100 * m[k]['throughput_ratio']:.1f}\\%"
        for k in ["relay_weight1024", "relay_activation1024", r["direct"]]
    )
    lines.append(f"{r['family']} & {r['task']} & {values} " + r"\\")
lines += [r"\bottomrule", r"\end{tabular}"]
Path("paper/iclr2027/generated/compression_objective_table.tex").write_text(
    "\n".join(lines) + "\n"
)
(root / "compression-objective-summary.json").write_text(
    json.dumps(
        dict(
            input_sha256=inputs,
            results=results,
            scope="32 exposed questions,128-token cap,rank1024 per arm. Posthoc arms inherit dense fitting and use512 calibration records; direct arms fit against the original teacher. Deployment rank is matched; total adaptation compute is not matched.",
        ),
        indent=2,
    )
    + "\n"
)
