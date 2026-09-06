"""Regenerate task-stratified paper evidence from the complete raw quality run."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("paper/iclr2027"))
    args = parser.parse_args()
    registry = Path("reports/mapper-scaling-20260905/task-complexity-results.json")
    result = json.loads(registry.read_text())
    with tempfile.TemporaryDirectory() as temporary:
        expected = Path(temporary) / "expected.json"
        subprocess.run(
            [
                sys.executable,
                "scripts/analyze_task_complexity.py",
                "--raw-root",
                str(args.raw_root),
                "--output",
                str(expected),
            ],
            check=True,
            capture_output=True,
        )
        if json.loads(expected.read_text()) != result:
            raise ValueError("task complexity evidence does not reproduce")
    labels = [
        ("native_ar", "AR"),
        ("native_target_dflash", "Native drafter"),
        ("optimized_source_reuse", "Source reuse"),
        ("relay_dense_n512", r"Dense, $N=512$"),
        ("relay_dense_n2048", r"Dense, $N=2048$"),
        ("relay_factorized1024_n512", r"Linear 1024, $N=512$"),
        ("relay_factorized4096_n2048", r"Linear 4096, $N=2048$"),
        ("relay_mlp4096_n2048", r"MLP 4096, $N=2048$"),
        ("relay_factorized512_n2048", "Linear 512"),
        ("relay_factorized512_n2048_continue32768", "Linear 512, continued"),
        ("relay_mlp512_n2048", "MLP 512"),
        ("relay_mlp512_n2048_continue32768", "MLP 512, continued"),
    ]
    if {name for name, _ in labels} != set(result["methods"]):
        raise ValueError("complexity table must include every evaluated method")
    groups = result["strata"]["difficulty"]
    names = ["level_1_2", "level_3", "level_4_5"]
    table = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        "Setting & "
        + " & ".join(
            f"{label} ($n={groups[name]['requests']}$)"
            for name, label in zip(
                names, ["Levels 1--2", "Level 3", "Levels 4--5"], strict=True
            )
        )
        + r" \\",
        r"\midrule",
    ]
    for method, label in labels:
        values = []
        for name in names:
            row = groups[name]["comparisons"]["relay_dense_n2048"]["methods"][method]
            lo, hi = [100 * x for x in row["throughput_ci95"]]
            values.append(f"{100 * row['throughput_ratio']:.1f} [{lo:.1f}, {hi:.1f}]")
        table.append(label + " & " + " & ".join(values) + r" \\")
    table += [r"\bottomrule", r"\end{tabular}"]
    generated = args.output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "task_difficulty_table.tex").write_text("\n".join(table) + "\n")
    (generated / "task_complexity_evidence_registry.json").write_text(
        json.dumps(
            {
                "raw_root": str(args.raw_root.resolve()),
                "input_sha256": {
                    str(registry): hashlib.sha256(registry.read_bytes()).hexdigest(),
                    **result["input_sha256"],
                },
                "scope": result["scope"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
