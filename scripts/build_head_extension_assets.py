"""Keep the broader head-precision result separate from its initial screen."""

import hashlib
import json
from pathlib import Path

from relayspec.ar_paper_evidence import summarize

root = Path("reports/autoresearch-20260907/run-28502")
inputs = {}
results = []
lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Workload & Head & Units & Native matches & Relay matches & AR capped \\",
         r"\midrule"]
for lane in range(4):
    status = json.loads((root / f"lane{lane}-status.json").read_text())
    if status["status"] != "pass":
        raise ValueError("Incomplete head-precision extension")
    path = root / f"lane{lane}/benchmark-rank0.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    result = summarize(rows, reference="native_ar")
    lookup = {(r["problem_id"], r["method"]): r for r in rows}
    ids = {r["problem_id"] for r in rows}
    native_relay_matches = sum(lookup[p, "native_target_dflash"]["output_hash"] ==
                               lookup[p, "relay_base"]["output_hash"] for p in ids)
    caps = {m: sum(r["output_tokens"] >= 512 for r in rows if r["method"] == m)
            for m in result["methods"]}
    task = "GSM8K" if lane < 2 else "MATH"
    head = "BF16" if lane % 2 == 0 else "FP32"
    n = result["requests"]
    native = result["methods"]["native_target_dflash"]["token_matches"]
    relay = result["methods"]["relay_base"]["token_matches"]
    results.append(dict(workload=task, head_precision=head, native_relay_matches=native_relay_matches,
                        capped=caps, **result))
    lines.append(f"{task} & {head} & {n} & {native}/{n} & {relay}/{n} & {caps['native_ar']}/{n} " + r"\\")
    inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(task, head, "native/relay matches", native_relay_matches, "of", n)
lines += [r"\bottomrule", r"\end{tabular}"]
Path("paper/iclr2027/generated/head_precision_extension_table.tex").write_text("\n".join(lines) + "\n")
Path("reports/autoresearch-20260907/head-extension-summary.json").write_text(json.dumps(
    dict(input_sha256=inputs, results=results,
         scope="Additional exposed development questions. Own-runtime AR matches; "
               "512-token cap. No full-answer MATH or general exactness claim. "
               "Native DFlash uses target output head for proposal logits too."), indent=2) + "\n")
