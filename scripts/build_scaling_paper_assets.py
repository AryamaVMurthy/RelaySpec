"""Render completed controlled fitting and paired code-quality evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("paper/iclr2027"))
    args = parser.parse_args()
    evidence = Path("reports/mapper-scaling-20260905")
    data_path = evidence / "historical-data-summary.json"
    quality_path = evidence / "code-quality-ar.json"
    data, quality = [json.loads(p.read_text()) for p in [data_path, quality_path]]
    if (
        not data["complete"]
        or len(data["fixed_data"]) != 7
        or len(data["continuous"]) != 5
    ):
        raise ValueError("controlled fitting curves are incomplete")
    inputs = {str(p): digest(p) for p in [data_path, quality_path]}
    for row in data["fixed_data"] + data["continuous"]:
        root = args.raw_root / row["path"]
        path = root / "analysis.json"
        analysis = json.loads(path.read_text())
        if analysis["status"] != "complete":
            raise ValueError("raw fitting result has not passed analysis")
        metrics = analysis["tasks"]["math500"]["methods"]["relay_p"]
        if (
            metrics["tokens_per_second"] != row["tokens_per_second"]
            or metrics["throughput_ratio"] != row["speedup_vs_ar"]
        ):
            raise ValueError("reported fitting throughput differs from source analysis")
        inputs[str(path)] = digest(path)
        for name, expected in analysis["raw_sha256"].items():
            path = root / name
            if digest(path) != expected:
                raise ValueError("raw fitting evidence hash mismatch")
            inputs[str(path)] = expected
    for name, expected in quality["source_sha256"].items():
        # The stored provenance records original absolute paths. Resolve the
        # stable reports-relative suffix against the requested evidence root.
        relative = "reports/" + name.split("/reports/", 1)[1]
        path = args.raw_root / relative
        if digest(path) != expected:
            raise ValueError("raw code-quality evidence hash mismatch")
        inputs[str(path)] = expected
    generated = args.output / "generated"
    figures = args.output / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.55), layout="constrained")
    for ax, rows, key, label in [
        (
            axes[0],
            data["fixed_data"],
            "distinct",
            "Distinct fitting records (1,024 updates)",
        ),
        (axes[1], data["continuous"], "updates", "Updates in one continuous fit"),
    ]:
        x = [r[key] for r in rows]
        y = [r["speedup_vs_ar"] for r in rows]
        errors = [
            [r["speedup_vs_ar"] - r["speedup_ci95"][0] for r in rows],
            [r["speedup_ci95"][1] - r["speedup_vs_ar"] for r in rows],
        ]
        ax.errorbar(
            x, y, yerr=errors, marker="o", color="#245988", capsize=3, linewidth=1.2
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [str(v) for v in x], rotation=35)
        ax.set_xlabel(label)
        ax.set_ylabel("End-to-end throughput / AR")
        ax.grid(alpha=0.2)
    fig.savefig(
        figures / "controlled_fitting.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(figures / "controlled_fitting.png", dpi=180)
    plt.close(fig)
    table = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Records & Tokens/s & $\times$ AR & Native retained & Best retained & Correct \\",
        r"\midrule",
    ]
    for row in data["fixed_data"]:
        table.append(
            f"{row['distinct']:,} & {row['tokens_per_second']:.2f} & {row['speedup_vs_ar']:.2f} & {100 * row['native_retained']:.1f}\\% & {100 * row['throughput_fraction_of_observed_best']:.1f}\\% & {row['correct']}/{row['requests']} "
            + r"\\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "controlled_data_table.tex").write_text("\n".join(table) + "\n")
    table = [r"\begin{tabular}{llrrrr}", r"\toprule"]
    table += [
        r"Target/task & Suite & AR & Relay & $\Delta$ (pp) & Paired 95\% interval \\",
        r"\midrule",
    ]
    for row in quality["comparisons"]:
        n = row["paired_tasks"]
        lo, hi = row["accuracy_delta_ci95"]
        task = "HumanEval" if row["benchmark"] == "humaneval" else "MBPP"
        table.append(
            f"{row['target_billions']}B {task} & {row['suite']} & {round(n * row['reference_accuracy'])}/{n} & {round(n * row['candidate_accuracy'])}/{n} & {100 * row['accuracy_delta']:+.2f} & $[{100 * lo:+.2f}, {100 * hi:+.2f}]$ "
            + r"\\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "eagle3_paired_code_quality.tex").write_text("\n".join(table) + "\n")
    (generated / "scaling_evidence_registry.json").write_text(
        json.dumps(
            {
                "raw_root": str(args.raw_root.resolve()),
                "input_sha256": inputs,
                "scope": "Completed historical-data and continuous fitting curves, plus saved-output AR-paired EAGLE-3 code quality. New Numina capacity cells are not included until complete.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
