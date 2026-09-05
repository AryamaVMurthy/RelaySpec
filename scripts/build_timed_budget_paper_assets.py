"""Regenerate matched warm-training tables from audited fitting and decoding."""

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
    registry = Path("reports/external-baselines-20260906/timed-budget-decoding.json")
    result = json.loads(registry.read_text())
    base = args.raw_root / "reports/mapper-scaling-20260905"
    with tempfile.TemporaryDirectory() as temporary:
        expected = Path(temporary) / "expected.json"
        subprocess.run(
            [
                sys.executable,
                "scripts/audit_timed_budget_decoding.py",
                "--fits",
                str(base / "timed-budget-pilot/run-27861"),
                str(base / "timed-budget-full/run-27864"),
                str(base / "timed-budget-full/run-27865"),
                "--run",
                str(base / "timed-budget-decoding/run-27866"),
                "--output",
                str(expected),
            ],
            check=True,
            capture_output=True,
        )
        if json.loads(expected.read_text()) != result:
            raise ValueError("timed-budget registry does not reproduce")
    table = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Budget & Objective & Updates & Warm s & Worker s & Tok/s & Feature ratio [95\% CI] \\",
        r"\midrule",
    ]
    labels = {
        "relay_feature": "Feature MSE",
        "relay_connector_ce": "Connector CE",
        "relay_lora32_s1729": "LoRA32 (1729)",
        "relay_lora32_s1730": "LoRA32 (1730)",
    }
    for multiplier in (0.25, 1, 4):
        records = [r for r in result["fitting"] if r["multiplier"] == multiplier]
        if len(records) != 4 or {
            r["total_distinct_training_records"] for r in records
        } != {512}:
            raise ValueError("timed table requires four 512-example fits per budget")
        for row in records:
            name = f"relay_budget{round(multiplier * 100):03d}_" + row[
                "method"
            ].removeprefix("relay_")
            values = result["matched_budget_against_feature"][str(multiplier)][
                "methods"
            ][name]
            lo, hi = values["throughput_ci95"]
            table.append(
                f"{multiplier:g}$\\times$ & {labels[row['method']]} & {row['updates']} & "
                f"{row['warm_training_seconds_with_initial_mapper']:.2f} & "
                f"{row['worker_seconds_with_initial_mapper']:.2f} & "
                f"{values['tokens_per_second']:.2f} & "
                f"{values['throughput_ratio']:.3f} [{lo:.3f}, {hi:.3f}] " + r"\\"
            )
        if multiplier != 4:
            table.append(r"\midrule")
    table += [r"\bottomrule", r"\end{tabular}"]
    controls = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Reference & Tok/s & Initial mapper ratio [95\% CI] \\",
        r"\midrule",
    ]
    for name, label in [
        ("native_ar", "AR"),
        ("native_target_dflash", "Native drafter"),
        ("optimized_source_reuse", "Source reuse"),
        ("relay_initial", "Initial mapper"),
        ("relay_dense8192_reference", "Dense, 8192 updates"),
    ]:
        values = result["against_initial"]["methods"][name]
        lo, hi = values["throughput_ci95"]
        controls.append(
            f"{label} & {values['tokens_per_second']:.2f} & "
            f"{values['throughput_ratio']:.3f} [{lo:.3f}, {hi:.3f}] " + r"\\"
        )
    controls += [r"\bottomrule", r"\end{tabular}"]
    generated = args.output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "timed_budget_table.tex").write_text("\n".join(table) + "\n")
    (generated / "timed_budget_controls_table.tex").write_text(
        "\n".join(controls) + "\n"
    )
    (generated / "timed_budget_evidence_registry.json").write_text(
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
