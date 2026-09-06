"""Render fixed research confirmation tables from completed paired analyses."""

import hashlib
import json
from pathlib import Path

out = Path("paper/iclr2027/generated")
inputs = {}
for job, filename, arms in [
    (
        28414,
        "interface_confirmation_table",
        [
            ("relay_base", "Five target layers", 52.43),
            ("relay_last2", "Two target layers", 20.97),
        ],
    ),
    (
        28416,
        "native_compression_confirmation_table",
        [
            ("native_target_dflash", "Released native", 83.89),
            ("relay_base", "Repacked native", 83.89),
            ("relay_svd1536", "Rank 1,536", 37.75),
        ],
    ),
]:
    path = Path(f"reports/autoresearch-20260907/run-{job}/confirmation-analysis.json")
    d = json.loads(path.read_text())
    if d["status"] != "complete" or d["summary"]["requests"] != 64:
        raise ValueError("Incomplete fixed comparison")
    inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    methods = d["summary"]["methods"]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Interface & Parameters & Tokens/s & Retention (95\% CI) & Correct \\",
        r"\midrule",
    ]
    for name, label, count in arms:
        r = methods[name]
        a, b = r["throughput_ci95"]
        lines.append(
            f"{label} & {count:.2f}M & {r['tokens_per_second']:.2f} & {100 * r['throughput_ratio']:.2f} [{100 * a:.2f}, {100 * b:.2f}]\\% & {round(r['accuracy'] * 64)}/64 "
            + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / f"{filename}.tex").write_text("\n".join(lines) + "\n")
(out / "interface_confirmation_provenance.json").write_text(
    json.dumps(
        dict(
            input_sha256=inputs,
            scope="Two fixed64-request comparisons; native comparison is secondary. Parameter counts concern only mapper/projection.",
        ),
        indent=2,
    )
    + "\n"
)
