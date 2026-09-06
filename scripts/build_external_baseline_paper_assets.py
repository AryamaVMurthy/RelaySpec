"""Rebuild separately scoped SD-square and PARD development tables from raw gates."""

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
    base = args.raw_root / "reports/mapper-scaling-20260905"
    registry_root = Path("reports/external-baselines-20260906")
    inputs, results = {}, {}
    with tempfile.TemporaryDirectory() as temporary:
        for name, script, parameters in [
            (
                "sd-square-decoding",
                "audit_sd_square_campaign.py",
                [
                    "--fits",
                    str(base / "sd-square-fullpilot/run-27888"),
                    str(base / "sd-square-rates/run-27890"),
                    "--runs",
                    str(base / "sd-square-shards/run-27900"),
                    str(base / "sd-square-shards/run-27901"),
                    "--ledger",
                    "reports/mapper-scaling-20260905/sd-square-shards/jobs.json",
                ],
            ),
            (
                "pard-decoding",
                "audit_pard_campaign.py",
                [
                    "--run",
                    str(base / "pard-campaign/run-27903"),
                    "--ledger",
                    "reports/mapper-scaling-20260905/pard-campaign/jobs.json",
                ],
            ),
        ]:
            expected = Path(temporary) / f"{name}.json"
            subprocess.run(
                [
                    sys.executable,
                    f"scripts/{script}",
                    *parameters,
                    "--output",
                    str(expected),
                ],
                check=True,
                capture_output=True,
            )
            registry = registry_root / f"{name}.json"
            result = json.loads(registry.read_text())
            if json.loads(expected.read_text()) != result:
                raise ValueError("external baseline registry does not reproduce")
            inputs[str(registry)] = hashlib.sha256(registry.read_bytes()).hexdigest()
            inputs.update(result["input_sha256"])
            results[name] = result
    sd = results["sd-square-decoding"]
    table = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Setting & Tok/s & AR ratio [95\% CI] & Independent ratio [95\% CI] & Exact AR \\",
        r"\midrule",
    ]
    labels = {
        "native_ar": "Matched AR",
        "sd2_independent": "Independent drafter",
        "sd2_zero_guidance": "Zero guidance",
    }
    for name, variant in sd["variants"].items():
        if variant["kind"] == "steering":
            rate = {
                4e-6: r"$4\times10^{-6}$",
                1e-5: r"$10^{-5}$",
                2e-5: r"$2\times10^{-5}$",
            }[variant["learning_rate"]]
            labels[name] = variant["objective"].upper() + ", " + rate
        row = sd["comparisons"]["native_ar"]["methods"][name]
        independent = sd["comparisons"]["sd2_independent"]["methods"][name]
        lo, hi = row["throughput_ci95"]
        ilo, ihi = independent["throughput_ci95"]
        table.append(
            f"{labels[name]} & {row['tokens_per_second']:.2f} & {row['throughput_ratio']:.3f} [{lo:.3f}, {hi:.3f}] & "
            f"{independent['throughput_ratio']:.3f} [{ilo:.3f}, {ihi:.3f}] & {sd['ar_exact_counts'][name]}/16 "
            + r"\\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    pard = results["pard-decoding"]
    ptable = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Setting & Tok/s & AR ratio [95\% CI] & Exact AR \\",
        r"\midrule",
    ]
    for name, label in [("native_ar", "Matched eager AR"), ("pard", "Released PARD")]:
        row = pard["comparison"]["methods"][name]
        lo, hi = row["throughput_ci95"]
        exact = 16 if name == "native_ar" else pard["ar_exact_count"]
        ptable.append(
            f"{label} & {row['tokens_per_second']:.2f} & {row['throughput_ratio']:.3f} [{lo:.3f}, {hi:.3f}] & {exact}/16 "
            + r"\\"
        )
    ptable += [r"\bottomrule", r"\end{tabular}"]
    generated = args.output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    for name, rows in [
        ("sd_square_development_table.tex", table),
        ("pard_development_table.tex", ptable),
    ]:
        (generated / name).write_text("\n".join(rows) + "\n")
    (generated / "external_baseline_evidence_registry.json").write_text(
        json.dumps(
            {
                "raw_root": str(args.raw_root.resolve()),
                "input_sha256": inputs,
                "scope": {name: result["scope"] for name, result in results.items()},
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
