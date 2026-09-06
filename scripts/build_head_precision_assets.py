"""Compare head-only precision with both its own AR and full-FP32 AR."""

import hashlib
import json
from pathlib import Path

from relayspec.ar_paper_evidence import summarize

root = Path("reports/autoresearch-20260907")
run = root / "run-28488"
inputs = {}
results = []
lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Target & Head & Own AR matches & FP32 AR matches & Tokens/s & Throughput / AR \\",
         r"\midrule"]
for lane in range(4):
    status_path = run / f"lane{lane}-status.json"
    if json.loads(status_path.read_text())["status"] != "pass":
        raise ValueError("Incomplete output-head diagnostic")
    path = run / f"lane{lane}/benchmark-rank0.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    metadata_path = run / f"lane{lane}/mapper-campaign.json"
    metadata = json.loads(metadata_path.read_text())
    promoted = lane % 2 == 1
    expected = ["torch.bfloat16", "torch.float32"] if promoted else ["torch.bfloat16"]
    if metadata["runtime_precision"] != "bfloat16" or metadata["target_parameter_dtypes"] != expected:
        raise ValueError("Unexpected actual target precision")
    if promoted and metadata["target_head_diagnostic"]["input_embedding_dtype"] != "torch.bfloat16":
        raise ValueError("Output-head intervention changed embedding precision")
    summary = summarize(rows, reference="native_ar")
    full_job = 28459 if lane < 2 else 28464
    full_paths = [root / f"run-{full_job}/lane{i}/benchmark-rank0.jsonl" for i in [1, 3]]
    full_rows = [json.loads(x) for p in full_paths for x in p.read_text().splitlines()]
    full_ar = {r["problem_id"]: r["output_hash"] for r in full_rows if r["method"] == "native_ar"}
    if set(full_ar) != {r["problem_id"] for r in rows} or summary["requests"] != 16:
        raise ValueError("FP32 reference uses different questions")
    fp32_matches = sum(r["output_hash"] == full_ar[r["problem_id"]]
                       for r in rows if r["method"] == "relay_base")
    model, head = ("0.6B" if lane < 2 else "8B"), ("FP32" if promoted else "BF16")
    m = summary["methods"]["relay_base"]
    lines.append(f"{model} & {head} & {m['token_matches']}/16 & {fp32_matches}/16 & "
                 f"{m['tokens_per_second']:.1f} & {m['throughput_ratio']:.2f} " + r"\\")
    results.append(dict(target=model, head_precision=head, full_fp32_ar_matches=fp32_matches,
                        runtime=metadata, **summary))
    for p in [status_path, path, metadata_path, *full_paths]:
        inputs[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
lines += [r"\bottomrule", r"\end{tabular}"]
Path("paper/iclr2027/generated/head_precision_table.tex").write_text("\n".join(lines) + "\n")
(root / "head-precision-summary.json").write_text(json.dumps(
    dict(input_sha256=inputs, results=results,
         scope="Sixteen exposed GSM8K questions per target. Same frozen mapper per "
               "target. Head-only promotion leaves embeddings/transformer/mapper BF16. "
               "Own AR reference uses the same head precision; full-FP32 AR is a separate "
               "execution policy. Cross-mode timings do not isolate kernel overhead."), indent=2) + "\n")
