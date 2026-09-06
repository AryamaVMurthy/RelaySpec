"""Render the complete smaller-verifier development comparison."""

import json
from pathlib import Path

result = json.loads(Path("reports/autoresearch-20260907/downscale-summary.json").read_text())
labels = {
    "native_ar": "AR",
    "optimized_source_reuse": "Source reuse",
    "relay_dense_a": "Dense, seed 1729",
    "relay_dense_b": "Dense, duplicate",
    "relay_dense_s1730": "Dense, seed 1730",
    "relay_factor1024": "Factored linear-1024",
    "relay_mlp1024": "MLP-1024",
}
lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Method & Tokens/s & Throughput / AR & AR time / time & Correct & AR matches \\",
         r"\midrule"]
for method, label in labels.items():
    m = result["summary"]["methods"][method]
    quality = result["quality"][method]
    lines.append(f"{label} & {m['tokens_per_second']:.1f} & {m['throughput_ratio']:.2f} & "
                 f"{m['request_time_ratio']:.2f} & {quality['correct']}/16 & {m['token_matches']}/16 " + r"\\")
lines += [r"\bottomrule", r"\end{tabular}"]
Path("paper/iclr2027/generated/downscale_table.tex").write_text("\n".join(lines) + "\n")
