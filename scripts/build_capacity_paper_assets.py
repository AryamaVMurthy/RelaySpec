"""Build source-checked capacity and continued-epoch figures for the appendix."""

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
    base = Path("reports/mapper-scaling-20260905")
    matrix_path = Path("configs/submission/scaling/matrix-focused-v1/matrix.json")
    fit_path, continued_path = (
        base / "primary-capacity-results.json",
        base / "continued-small-data-results.json",
    )
    matrix, fits, continued = [
        json.loads(p.read_text()) for p in [matrix_path, fit_path, continued_path]
    ]
    if fits["matrix_sha256"] != digest(matrix_path):
        raise ValueError("fitting registry has the wrong matrix")
    primary = matrix["primary_cells"]
    if len(primary) != 30 or any(t["name"] not in fits["results"] for t in primary):
        raise ValueError(
            "complete primary capacity matrix is required for paper assets"
        )
    inputs = {str(p): digest(p) for p in [matrix_path, fit_path, continued_path]}
    for t in primary:
        r = fits["results"][t["name"]]
        expected_parameters = (
            20480 * 2560 if t["architecture"] == "dense" else t["width"] * 23040
        )
        if r["parameters"] != expected_parameters or t["validation_records"] != 1024:
            raise ValueError("paper capacity dimensions or validation budget changed")
        folder = args.raw_root / r["raw_directory"]
        for p, sha in [
            (folder / "fit-complete.json", r["fit_sha256"]),
            (folder / "validation.jsonl", r["validation_sha256"]),
            (folder.parent.parent / "batch-gate.json", r["batch_gate_sha256"]),
        ]:
            if digest(p) != sha:
                raise ValueError(f"capacity source hash mismatch: {p}")
            inputs[str(p)] = sha
    if continued["status"] != "complete" or len(continued["results"]) != 4:
        raise ValueError("all four continuation trajectories are required")
    for r in continued["results"]:
        for name, sha in r["source_sha256"].items():
            p = args.raw_root / base / name
            if digest(p) != sha:
                raise ValueError(f"continuation source hash mismatch: {p}")
            inputs[str(p)] = sha
    figures, generated = args.output / "figures", args.output / "generated"
    figures.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    styles = [
        ("factorized", "Factored linear", "#245988", "o"),
        ("mlp", "GELU MLP", "#b65b20", "^"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7), layout="constrained", sharey=True)
    for ax, n in zip(axes, [512, 2048], strict=True):
        for family, label, color, marker in styles:
            values = [
                fits["results"][f"{family}{w}-n{n}-s1729"]
                for w in [64, 128, 256, 512, 1024, 2048, 4096]
            ]
            ax.plot(
                [v["parameters"] / 1e6 for v in values],
                [v["validation_objective"] for v in values],
                marker=marker,
                color=color,
                label=label,
                linestyle="-" if family == "factorized" else "--",
                linewidth=1.2,
                markersize=4,
            )
        dense = fits["results"][f"dense-n{n}-s1729"]
        ax.scatter(
            [dense["parameters"] / 1e6],
            [dense["validation_objective"]],
            marker="*",
            s=90,
            color="black",
            label="Single linear",
            zorder=4,
        )
        ax.set_xscale("log", base=2)
        ticks = [w * 23040 / 1e6 for w in [64, 256, 1024, 4096]]
        ax.set_xticks(ticks, [f"{x:.1f}" for x in ticks])
        ax.set_xlabel("Fitted parameters (millions)")
        ax.set_title(f"{n:,} distinct fitting records")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Validation relative interface MSE")
    axes[1].legend(fontsize=7.5)
    for suffix in ["pdf", "png"]:
        opts = (
            {"metadata": {"CreationDate": None, "ModDate": None}}
            if suffix == "pdf"
            else {"dpi": 180}
        )
        fig.savefig(figures / f"numina_capacity.{suffix}", **opts)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7), layout="constrained", sharey=True)
    for ax, n in zip(axes, [512, 2048], strict=True):
        for family, label, color, marker in styles:
            r = next(
                r
                for r in continued["results"]
                if r["trial"]["architecture"] == family
                and r["trial"]["distinct_examples"] == n
            )
            points = [p for p in r["validation_trajectory"] if p["step"] >= 2048]
            for split, line in [("train", "--"), ("validation", "-")]:
                ax.plot(
                    [p["step"] for p in points],
                    [p["groups"][split]["objective"] for p in points],
                    color=color,
                    linestyle=line,
                    marker=marker if split == "validation" else None,
                    label=f"{label}: {split}",
                    markersize=4,
                    linewidth=1.2,
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks(
            [2048, 4096, 8192, 16384, 32768],
            ["2,048", "4,096", "8,192", "16,384", "32,768"],
            rotation=25,
        )
        ax.set_title(f"Width 512; {n:,} fitting records")
        ax.set_xlabel("Total optimizer updates")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Relative interface MSE")
    axes[1].legend(fontsize=7, loc="upper right")
    for suffix in ["pdf", "png"]:
        opts = (
            {"metadata": {"CreationDate": None, "ModDate": None}}
            if suffix == "pdf"
            else {"dpi": 180}
        )
        fig.savefig(figures / f"numina_epochs.{suffix}", **opts)
    plt.close(fig)
    table = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Width & Params (M) & Linear, $N{=}512$ & MLP, $N{=}512$ & Linear, $N{=}2048$ & MLP, $N{=}2048$ \\",
        r"\midrule",
    ]
    for width in [64, 128, 256, 512, 1024, 2048, 4096]:
        values = [
            fits["results"][f"{family}{width}-n{n}-s1729"]["validation_objective"]
            for n in [512, 2048]
            for family in ["factorized", "mlp"]
        ]
        table.append(
            f"{width} & {width * 23040 / 1e6:.2f} & "
            + " & ".join(f"{v:.4f}" for v in values)
            + r" \\"
        )
    table += [
        r"\midrule",
        f"Dense & 52.43 & {fits['results']['dense-n512-s1729']['validation_objective']:.4f} & --- & {fits['results']['dense-n2048-s1729']['validation_objective']:.4f} & --- "
        + r"\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (generated / "numina_capacity_table.tex").write_text("\n".join(table) + "\n")
    (generated / "capacity_evidence_registry.json").write_text(
        json.dumps(
            {
                "raw_root": str(args.raw_root.resolve()),
                "input_sha256": inputs,
                "scope": "All30 predeclared primary capacity fits at8192updates and four continued width512 trajectories at32768 total updates. N512/2048 only. Single-seed fitting diagnostics; task quality, decoding and penalty sweeps are separate.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
