"""Regenerate the capped development-quality table from fully paired raw rows."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("paper/iclr2027"))
    args = parser.parse_args()
    registry = Path("reports/mapper-scaling-20260905/small-data-quality-results.json")
    result = json.loads(registry.read_text())
    run = (
        args.raw_root
        / "reports/mapper-scaling-20260905/small-data-quality-full/run-27806"
    )
    if Path(result["run"]).resolve() != run.resolve():
        raise ValueError("quality registry refers to another raw run")
    with tempfile.TemporaryDirectory() as tmp:
        expected = Path(tmp) / "expected.json"
        subprocess.run(
            [
                sys.executable,
                "scripts/summarize_small_quality.py",
                "--run",
                str(run),
                "--output",
                str(expected),
            ],
            check=True,
            capture_output=True,
        )
        if json.loads(expected.read_text()) != result:
            raise ValueError("quality registry does not reproduce from raw/scored rows")
    labels = [
        ("native_ar", "AR", "--", "--"),
        ("native_target_dflash", "Native drafter", "--", "--"),
        ("optimized_source_reuse", "Source reuse", "--", "--"),
        ("relay_dense_n512", "Dense", "512", "8192"),
        ("relay_dense_n2048", "Dense", "2048", "8192"),
        ("relay_factorized1024_n512", "Linear 1024", "512", "8192"),
        ("relay_factorized4096_n2048", "Linear 4096", "2048", "8192"),
        ("relay_mlp4096_n2048", "MLP 4096", "2048", "8192"),
        ("relay_factorized512_n2048", "Linear 512", "2048", "8192"),
        ("relay_factorized512_n2048_continue32768", "Linear 512", "2048", "32768"),
        ("relay_mlp512_n2048", "MLP 512", "2048", "8192"),
        ("relay_mlp512_n2048_continue32768", "MLP 512", "2048", "32768"),
    ]
    if {n for n, *_ in labels} != set(result["against_ar"]["methods"]):
        raise ValueError("quality table omits or adds an evaluated method")
    table = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & $N$ & Updates & Tok/s & Dense retained (\%) [95\% CI] & Correct & Cap \\",
        r"\midrule",
    ]
    for name, label, n, steps in labels:
        values = result["against_ar"]["methods"][name]
        dense = result["against_dense_n2048"]["methods"][name]
        lo, hi = [100 * x for x in dense["throughput_ci95"]]
        table.append(
            f"{label} & {n} & {steps} & {values['tokens_per_second']:.2f} & "
            f"{100 * dense['throughput_ratio']:.2f} [{lo:.2f}, {hi:.2f}] & "
            f"{round(values['accuracy'] * 128)}/128 & {result['cap_length_output_counts'][name]} "
            + r"\\"
        )
        if name == "optimized_source_reuse":
            table.append(r"\midrule")
    table += [r"\bottomrule", r"\end{tabular}"]
    generated = args.output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "small_quality_table.tex").write_text("\n".join(table) + "\n")
    (generated / "small_quality_evidence_registry.json").write_text(
        json.dumps(
            {
                "raw_root": str(args.raw_root.resolve()),
                "input_sha256": {
                    str(registry): digest(registry),
                    str(run / "analysis.json"): result["analysis_sha256"],
                    str(run / "math-scored.jsonl"): result["scored_rows_sha256"],
                    **{
                        str(run / name): sha
                        for name, sha in result["raw_sha256"].items()
                    },
                },
                "scope": result["scope"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
