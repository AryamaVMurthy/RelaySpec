"""Reconstruct paired compact-interface composition screens, including uncertainty."""

import hashlib
import json
from pathlib import Path

from relayspec.ar_paper_evidence import summarize

root = Path("reports/autoresearch-20260907")
inputs = {}
results = []
for job, tasks in [
    (28431, ["MATH", "Dialogue", "MATH", "Dialogue"]),
    (28432, ["Code", "GSM8K", "Code", "GSM8K"]),
]:
    for lane, task in enumerate(tasks):
        run = root / f"run-{job}"
        status = json.loads((run / f"lane{lane}-status.json").read_text())
        if status["status"] != "pass":
            raise ValueError("Incomplete composition screen")
        path = run / f"lane{lane}" / "benchmark-rank0.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        result = summarize(rows, reference="relay_math")
        results.append(dict(interface="Retargeted" if lane < 2 else "Native",
                            workload=task, job=job, lane=lane, **result))
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

lines = [r"\begin{tabular}{llrr}", r"\toprule",
         r"Interface & Workload & Units & Mixed / math throughput (95\% CI) \\",
         r"\midrule"]
for result in sorted(results, key=lambda r: (r["interface"], r["workload"])):
    m = result["methods"]["relay_mixed"]
    low, high = m["throughput_ci95"]
    lines.append(
        f"{result['interface']} & {result['workload']} & {result['clusters']} & "
        f"{100*m['throughput_ratio']:.1f}\\% [{100*low:.1f}, {100*high:.1f}] " + r"\\"
    )
lines += [r"\bottomrule", r"\end{tabular}"]
Path("paper/iclr2027/generated/compact_composition_table.tex").write_text("\n".join(lines) + "\n")
(root / "compact-composition-summary.json").write_text(json.dumps(dict(
    input_sha256=inputs, results=results,
    scope="Exposed development screens; seed 1729; 512 fitting records and 8192 updates. "
          "Paired request bootstrap, clustered by conversation for dialogue. Intervals "
          "exclude fitting-seed and selection uncertainty. Equal records, not equal tokens."
), indent=2) + "\n")
