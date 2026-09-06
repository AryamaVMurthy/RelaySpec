"""Compare native interface students with a released-native reference on each workload."""

import hashlib
import json
from pathlib import Path

from relayspec.ar_paper_evidence import summarize

root = Path("reports/autoresearch-20260907/run-28429")
inputs = {}
results = []
for lane, task in enumerate(["GSM8K", "MATH", "Code", "Dialogue"]):
    path = root / f"lane{lane}" / "benchmark-rank0.jsonl"
    status = json.loads((root / f"lane{lane}-status.json").read_text())
    if status["status"] != "pass":
        raise ValueError("Incomplete native-student screen")
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    result = summarize(rows, reference="native_target_dflash")
    results.append(dict(task=task, **result))
    inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
lines = [
    r"\begin{tabular}{lrrrr}",
    r"\toprule",
    r"Workload & Units & One layer & Two layers & SVD rank 1,536 \\",
    r"\midrule",
]
for r in results:
    values = " & ".join(
        f"{100 * r['methods'][m]['throughput_ratio']:.1f}\\%"
        for m in ["relay_one", "relay_two", "relay_svd1536"]
    )
    lines.append(f"{r['task']} & {r['clusters']} & {values} " + r"\\")
lines += [r"\bottomrule", r"\end{tabular}"]
Path("paper/iclr2027/generated/native_student_table.tex").write_text(
    "\n".join(lines) + "\n"
)
Path("reports/autoresearch-20260907/native-student-summary.json").write_text(
    json.dumps(
        dict(
            input_sha256=inputs,
            results=results,
            scope="Exposed development screens. Native reference is the released implementation, not repacked FC. Units are questions or whole dialogue conversations. No code-test or dialogue-quality claim.",
        ),
        indent=2,
    )
    + "\n"
)
