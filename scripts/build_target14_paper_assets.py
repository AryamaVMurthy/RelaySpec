"""Reproduce both 14B fitting and capped decoding studies before paper export."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from relayspec.cached_fit_evidence import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("paper/iclr2027"))
    args = parser.parse_args()
    base = Path("reports/mapper-scaling-20260905")
    generated, figures = args.output / "generated", args.output / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for family, group in [("dflash", "dflash"), ("eagle3", "eagle")]:
        fit_path = base / f"target14b-{family}-fit-results.json"
        decode_path = base / f"target14b-{family}-capacity-results.json"
        fits, decoded = [json.loads(p.read_text()) for p in [fit_path, decode_path]]
        ledger_path = base / f"target14b-capacity-{group}/jobs.json"
        ledger = json.loads(ledger_path.read_text())
        if len(ledger["jobs"]) != 1:
            raise ValueError("14B capacity requires its single complete campaign")
        run = args.raw_root / ledger["jobs"][0]["local"]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for script, arguments, expected in [
                ("audit_target14_fits.py", ["--raw-root", str(args.raw_root)], fits),
                (
                    "audit_target14_capacity.py",
                    [
                        "--run",
                        str(run),
                        "--ledger",
                        str(ledger_path),
                        "--scoring-repo",
                        str(args.raw_root),
                    ],
                    decoded,
                ),
            ]:
                result_path = directory / "audit.json"
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/" + script,
                        "--family",
                        family,
                        *arguments,
                        "--output",
                        str(result_path),
                    ],
                    check=True,
                    capture_output=True,
                )
                if json.loads(result_path.read_text()) != expected:
                    raise ValueError(f"{family} evidence does not reproduce: {script}")
            subprocess.run(
                [
                    sys.executable,
                    "scripts/summarize_target14_fitting.py",
                    "--family",
                    family,
                    "--output",
                    str(directory / "fitting"),
                ],
                check=True,
                capture_output=True,
            )
            for extension in ["pdf", "png"]:
                shutil.copyfile(
                    directory / "fitting" / f"validation-trajectories.{extension}",
                    figures / f"target14_{family}_fitting.{extension}",
                )
        for path, result in [(fit_path, fits), (decode_path, decoded)]:
            inputs[str(path)] = digest(path)
            inputs.update(result["input_sha256"])
        reference = f"relay_{family}14_dense_n2048"
        comparisons = decoded["comparisons"][reference]["methods"]
        table = [
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Mapper & $N$ & Params (M) & Train / val. & Tokens/s & Retained (\%) \\",
            r"\midrule",
        ]
        records = sorted(
            fits["results"].values(),
            key=lambda r: (
                r["trial"]["distinct_examples"],
                {"dense": 0, "factorized": 1, "mlp": 2}[r["trial"]["architecture"]],
                r["trial"].get("width", 0),
                r["trial"]["seed"],
            ),
        )
        for record in records:
            trial = record["trial"]
            matches = [
                name
                for name, variant in decoded["variants"].items()
                if variant.get("trial") == trial
            ]
            if len(matches) != 1:
                raise ValueError(f"missing unique decoding binding for {trial['name']}")
            row = comparisons[matches[0]]
            label = {"dense": "Dense", "factorized": "Linear", "mlp": "MLP"}[
                trial["architecture"]
            ]
            if "width" in trial:
                label += f" {trial['width']}"
            if trial["seed"] != 1729:
                label += " (seed 1730)"
            lo, hi = [100 * x for x in row["throughput_ci95"]]
            table.append(
                f"{label} & {trial['distinct_examples']} & {record['parameters'] / 1e6:.2f} & "
                f"{record['train_objective']:.3f} / {record['validation_objective']:.3f} & "
                f"{row['tokens_per_second']:.2f} & {100 * row['throughput_ratio']:.1f} [{lo:.1f}, {hi:.1f}]"
                + r" \\"
            )
        table += [r"\bottomrule", r"\end{tabular}"]
        (generated / f"target14_{family}_table.tex").write_text("\n".join(table) + "\n")
    (generated / "target14_evidence_registry.json").write_text(
        json.dumps(
            {
                "raw_root": str(args.raw_root.resolve()),
                "input_sha256": inputs,
                "scope": "Both complete 16-fit matrices and their fixed 8192-update endpoints. Sixteen exposed requests at a 256-token cap per family. Request intervals exclude fitting-seed and selection uncertainty. No full-answer quality or cross-runtime ranking claim.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
