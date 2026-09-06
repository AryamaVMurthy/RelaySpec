"""Report precision diagnostics across all tested smaller-verifier workloads."""

import hashlib
import json
from pathlib import Path

import yaml

from relayspec.ar_paper_evidence import summarize

root = Path("reports/autoresearch-20260907")
inputs = {}
results = []
for job, workload, precision, lanes in [
    (28459, "GSM8K", "BF16", [0, 2]),
    (28459, "GSM8K", "FP32", [1, 3]),
    (28460, "Code", "BF16", [0]),
    (28460, "Code", "FP32", [1]),
    (28460, "Dialogue", "BF16", [2]),
    (28460, "Dialogue", "FP32", [3]),
]:
    rows = []
    for lane in lanes:
        folder = root / f"run-{job}"
        status_path = folder / f"lane{lane}-status.json"
        if json.loads(status_path.read_text())["status"] != "pass":
            raise ValueError("Incomplete precision breadth screen")
        path = folder / f"lane{lane}/benchmark-rank0.jsonl"
        config_path = folder / f"lane{lane}/config.yaml"
        config = yaml.safe_load(config_path.read_text())
        expected_precision = "bfloat16" if precision == "BF16" else "float32"
        if config["benchmark"]["precision"] != expected_precision:
            raise ValueError("Precision differs from declaration")
        rows.extend(json.loads(x) for x in path.read_text().splitlines())
        for p in [path, config_path, status_path]:
            inputs[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    summary = summarize(rows, reference="native_ar")
    results.append(dict(workload=workload, precision=precision, **summary))
lines = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"Workload & Precision & Units & AR matches & Throughput / AR (95\% CI) \\",
         r"\midrule"]
for result in results:
    m = result["methods"]["relay_base"]
    low, high = m["throughput_ci95"]
    lines.append(f"{result['workload']} & {result['precision']} & {result['clusters']} & "
                 f"{m['token_matches']}/{result['requests']} & "
                 f"{m['throughput_ratio']:.2f} [{low:.2f}, {high:.2f}] " + r"\\")
lines += [r"\bottomrule", r"\end{tabular}"]
Path("paper/iclr2027/generated/precision_breadth_table.tex").write_text("\n".join(lines) + "\n")
(root / "precision-breadth-summary.json").write_text(json.dumps(
    dict(input_sha256=inputs, results=results,
         scope="Qwen3-0.6B precision diagnostics. Paired intervals compare AR within "
               "each precision and resample whole conversations for dialogue. Code "
               "and GSM8K have512-token caps, dialogue256 per turn. Exact output "
               "agreement is separate from task quality. No untouched confirmation."), indent=2) + "\n")
