"""Regenerate all penalty evidence from the completed small-data fit registry."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from summarize_focused_scaling import collect, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("paper/iclr2027"))
    args = parser.parse_args()
    matrix = Path("configs/submission/scaling/matrix-focused-v1/matrix.json")
    registry = Path("reports/mapper-scaling-20260905/focused-results.json")
    fits = collect(args.raw_root, matrix)
    if fits != json.loads(registry.read_text()) or fits["status"] != "complete":
        raise ValueError("complete current source-verified fitting registry required")
    results = fits["results"]
    penalties = [r for r in results.values() if r["trial"]["study"] == "regularization"]
    if len(penalties) != 42:
        raise ValueError("all 42 declared nonzero penalty cells are required")
    inputs = {str(matrix): digest(matrix), str(registry): digest(registry)}
    for result in results.values():
        folder = args.raw_root / result["raw_directory"]
        for path, sha in [
            (folder / "fit-complete.json", result["fit_sha256"]),
            (folder / "validation.jsonl", result["validation_sha256"]),
            (folder.parent.parent / "batch-gate.json", result["batch_gate_sha256"]),
        ]:
            if digest(path) != sha:
                raise ValueError("raw fitting evidence hash changed")
            inputs[str(path)] = sha
    figures, generated = args.output / "figures", args.output / "generated"
    figures.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(7, 4), layout="constrained", sharey=True)
    families = [
        ("dense", "Dense"),
        ("factorized512", "Factored linear 512"),
        ("mlp512", "GELU MLP 512"),
    ]
    table = [
        r"\begin{tabular}{rlrrr}",
        r"\toprule",
        r"$N$ & Mapper & No penalty & Best tested L2 & Best tested AdamW \\",
        r"\midrule",
    ]
    contrasts = []
    for row, n in enumerate((512, 2048)):
        for column, (family, label) in enumerate(families):
            stem = f"{family}-n{n}-s1729"
            baseline = results[stem]["validation_objective"]
            ax = axes[row, column]
            best = []
            for suffix, key, name, color, marker, count in [
                ("l2", "l2_weight", "L2", "#245988", "o", 4),
                ("wd", "weight_decay", "AdamW", "#b65b20", "^", 3),
            ]:
                cells = sorted(
                    [
                        r
                        for name_, r in results.items()
                        if name_.startswith(stem + f"-{suffix}-")
                    ],
                    key=lambda r: r["trial"][key],
                )
                if len(cells) != count:
                    raise ValueError("incomplete penalty group")
                ax.plot(
                    [r["trial"][key] for r in cells],
                    [100 * (r["validation_objective"] / baseline - 1) for r in cells],
                    color=color,
                    marker=marker,
                    label=name,
                    linewidth=1.2,
                    markersize=4,
                )
                winner = min(cells, key=lambda r: r["validation_objective"])
                best.append(winner["validation_objective"])
                contrasts.append(
                    {
                        "baseline_trial": stem,
                        "baseline_validation": baseline,
                        "penalty": name,
                        "best_development_trial": winner["trial"]["name"],
                        "coefficient": winner["trial"][key],
                        "validation": winner["validation_objective"],
                        "relative_change_percent": 100
                        * (winner["validation_objective"] / baseline - 1),
                        "tested_trials": [r["trial"]["name"] for r in cells],
                    }
                )
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xscale("log")
            ax.set_xticks([1e-7, 1e-4, 1e-2])
            ax.set_title(f"{label}, N={n:,}")
            ax.grid(alpha=0.2)
            if row == 1:
                ax.set_xlabel("Penalty coefficient")
            if column == 0:
                ax.set_ylabel("Validation change (%)")
            table.append(
                f"{n} & {label} & {baseline:.5f} & {best[0]:.5f} & {best[1]:.5f} "
                + r"\\"
            )
        if row == 0:
            table.append(r"\midrule")
    axes[0, 2].legend(fontsize=7)
    for suffix in ("pdf", "png"):
        options = (
            {"metadata": {"CreationDate": None, "ModDate": None}}
            if suffix == "pdf"
            else {"dpi": 180}
        )
        fig.savefig(figures / f"numina_regularization.{suffix}", **options)
    plt.close(fig)
    table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "numina_regularization_table.tex").write_text("\n".join(table) + "\n")
    (generated / "regularization_evidence_registry.json").write_text(
        json.dumps(
            {
                "raw_root": str(args.raw_root.resolve()),
                "input_sha256": inputs,
                "contrasts": contrasts,
                "scope": "All 42 nonzero penalty cells plus matched zero controls, N512/2048, 8192 updates, validation1024, seed1729. L2 and AdamW are separate arms. Best coefficients are selected on fitting validation; no seed significance, decoding improvement or task-quality claim follows.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
