"""Regenerate EAGLE small-data fitting endpoints and early-stopping evidence."""

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
    matrix = Path("configs/submission/scaling/matrix-eagle3-small-v1/matrix.json")
    registry = Path("reports/mapper-scaling-20260905/eagle3-capacity-results.json")
    fits = collect(args.raw_root, matrix)
    if (
        fits != json.loads(registry.read_text())
        or fits["status"] != "complete"
        or fits["completed_cells"] != 16
    ):
        raise ValueError("all16 source-verified EAGLE fits are required")
    results = fits["results"]
    inputs = {str(matrix): digest(matrix), str(registry): digest(registry)}
    selected = {}
    for name, result in results.items():
        folder = args.raw_root / result["raw_directory"]
        for path, sha in [
            (folder / "fit-complete.json", result["fit_sha256"]),
            (folder / "validation.jsonl", result["validation_sha256"]),
            (folder.parent.parent / "batch-gate.json", result["batch_gate_sha256"]),
        ]:
            if digest(path) != sha:
                raise ValueError("EAGLE raw evidence changed")
            inputs[str(path)] = sha
        best = min(
            result["validation_trajectory"],
            key=lambda p: p["groups"]["validation"]["objective"],
        )
        selected[name] = {
            "step": best["step"],
            "validation": best["groups"]["validation"]["objective"],
            "endpoint": result["validation_objective"],
        }
    figures, generated = args.output / "figures", args.output / "generated"
    figures.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    families = [("dense", "Dense")] + [
        (f"{f}{w}", f"{'Linear' if f == 'factorized' else 'MLP'} {w}")
        for f in ("factorized", "mlp")
        for w in (128, 512, 2048)
    ]
    table = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{$N=512$} & \multicolumn{3}{c}{$N=2048$} \\",
        r"Mapper & M weights & End & Best & Update & End & Best & Update \\",
        r"\midrule",
    ]
    for family, label in families:
        points = [selected[f"eagle3-{family}-n{n}-s1729"] for n in (512, 2048)]
        parameters = results[f"eagle3-{family}-n512-s1729"]["parameters"]
        table.append(
            f"{label} & {parameters / 1e6:.2f} & "
            + " & ".join(
                f"{p['endpoint']:.5f} & {p['validation']:.5f} & {p['step']}"
                for p in points
            )
            + r" \\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "eagle_small_capacity_table.tex").write_text("\n".join(table) + "\n")
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(
        1, 2, figsize=(6.6, 2.45), layout="constrained", sharey=True
    )
    for ax, n in zip(axes, (512, 2048), strict=True):
        for family, label, color, marker in [
            ("dense", "Dense", "#242424", "o"),
            ("factorized2048", "Linear 2048", "#245988", "s"),
            ("mlp2048", "MLP 2048", "#b65b20", "^"),
        ]:
            result = results[f"eagle3-{family}-n{n}-s1729"]
            points = [p for p in result["validation_trajectory"] if p["step"] > 0]
            ax.plot(
                [p["step"] for p in points],
                [p["groups"]["validation"]["objective"] for p in points],
                color=color,
                marker=marker,
                markersize=3,
                linewidth=1.1,
                label=label,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([128, 512, 2048, 8192], ["128", "512", "2048", "8192"])
        ax.set_title(f"EAGLE-3, N={n:,}")
        ax.set_xlabel("Mapper updates")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Validation relative interface MSE")
    axes[1].legend(fontsize=7)
    for suffix in ("pdf", "png"):
        options = (
            {"metadata": {"CreationDate": None, "ModDate": None}}
            if suffix == "pdf"
            else {"dpi": 180}
        )
        fig.savefig(figures / f"eagle_small_epochs.{suffix}", **options)
    plt.close(fig)
    (generated / "eagle_capacity_evidence_registry.json").write_text(
        json.dumps(
            {
                "raw_root": str(args.raw_root.resolve()),
                "input_sha256": inputs,
                "selected": selected,
                "scope": "Completed EAGLE-3/Qwen3-8B small-data fitting study.14 primary cells "
                "and two additional denseN2048 seeds. Relative interface MSE with "
                "the EAGLE input-normalization convention. N512/2048,8192 updates, "
                "1024 validation records. Best checkpoints are feature-development "
                "selections. No decoding or task-quality conclusion follows.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
