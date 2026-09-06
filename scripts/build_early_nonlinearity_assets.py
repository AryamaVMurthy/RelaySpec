"""Connect early-layer fitting quality and decoding at matched mapper size."""

import hashlib
import json
from pathlib import Path

from relayspec.ar_paper_evidence import summarize

root = Path("reports/autoresearch-20260907/run-28435")
inputs = {}
results = []
lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Family & Early-layer map & Train error & Val. error & Progress/cycle & Retention \\",
         r"\midrule"]
for lane in range(4):
    status = json.loads((root / f"lane{lane}-status.json").read_text())
    if status["status"] != "pass":
        raise ValueError("Incomplete early-layer screen")
    path = root / f"lane{lane}" / "benchmark-rank0.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    summary = summarize(rows, reference="relay_base")
    validation_path = next((root / f"lane{lane}/fitting").glob("*/validation.jsonl"))
    completion_path = validation_path.parent / "fit-complete.json"
    validation = json.loads(validation_path.read_text().splitlines()[-1])
    complete = json.loads(completion_path.read_text())
    if complete["parameters"] != 20966400 or complete["steps"] != 8192:
        raise ValueError("Parameter or fitting-budget mismatch")
    if validation["step"] != 8192 or complete["distinct_records_seen"] != 512:
        raise ValueError("Fitting endpoint or record mismatch")
    family = "DFlash" if lane < 2 else "EAGLE-3"
    architecture = "Factored linear" if lane % 2 == 0 else "GELU MLP"
    train = validation["groups"]["train"]["relative_mse"]
    val = validation["groups"]["validation"]["relative_mse"]
    m = summary["methods"]["relay_reduced"]
    lines.append(f"{family} & {architecture} & {train:.3f} & {val:.3f} & "
                 f"{m['progress_per_cycle']:.2f} & {100*m['throughput_ratio']:.1f}\\% " + r"\\")
    results.append(dict(family=family, architecture=architecture, training_error=train,
                        validation_error=val, parameters=complete["parameters"], **summary))
    for p in [path, validation_path, completion_path]:
        inputs[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
lines += [r"\bottomrule", r"\end{tabular}"]
Path("paper/iclr2027/generated/early_nonlinearity_table.tex").write_text("\n".join(lines) + "\n")
Path("reports/autoresearch-20260907/early-nonlinearity-summary.json").write_text(json.dumps(
    dict(input_sha256=inputs, results=results,
         scope="Eight exposed GSM8K requests with128-token cap. Width1950, layers[1,9], "
               "512 records,8192 updates,seed1729. Exactly matched parameter count for "
               "MLP/factorized maps. Each arm has its own paired full, early-dense and "
               "late-dense controls. Negative results do not establish unrecoverability."),
    indent=2) + "\n")
