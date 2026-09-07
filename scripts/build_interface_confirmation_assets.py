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
    (
        28524,
        "native_student_confirmation_table",
        [
            ("native_target_dflash", "Released native", 83.89),
            ("relay_base", "Repacked native", 83.89),
            ("relay_two", "Two-layer student", 33.55),
            ("relay_one", "One-layer student", 16.78),
            ("relay_svd1536", "Rank 1,536", 37.75),
        ],
    ),
    (
        28543,
        "native_short_confirmation_table",
        [
            ("native_target_dflash", "Released native", 83.89),
            ("relay_base", "Repacked native", 83.89),
            ("relay_cropped", "Untrained crop", 33.55),
            ("relay_short16", "16 records, 128 updates", 33.55),
            ("relay_short128", "128 records, 128 updates", 33.55),
        ],
    ),
    (
        28560,
        "native_eagle_confirmation_table",
        [
            ("native_target_eagle3", "Released native", 83.89),
            ("relay_base", "Repacked native", 83.89),
            ("relay_short", "128-update inherited", 33.55),
            ("relay_two", "8,192-update inherited", 33.55),
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
            scope="Initial retargeting primary and native-SVD secondary comparisons share64 questions; native-student primary and short-native primary each use a further distinct64-question set. Short-native primary is16 records and fails the fixed retention criterion;128 records remains descriptive. Native EAGLE primary uses the final distinct64 reserve questions. All256 reserve questions are consumed. Parameter counts concern only mapper/projection.",
        ),
        indent=2,
    )
    + "\n"
)
