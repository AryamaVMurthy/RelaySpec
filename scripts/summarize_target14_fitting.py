"""Render complete 14B fitting trajectories without implying decoding rankings."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from relayspec.cached_fit_evidence import digest


def label(trial):
    return (
        "Dense"
        if trial["architecture"] == "dense"
        else f"{'MLP' if trial['architecture'] == 'mlp' else 'Factorized'} {trial['width']}"
    )


def style(trial):
    architecture = trial["architecture"]
    return {
        "color": {"dense": "#222222", "factorized": "#0072B2", "mlp": "#D55E00"}[
            architecture
        ],
        "linestyle": "-"
        if architecture == "dense"
        else {512: ":", 1024: "--", 4096: "-"}[trial["width"]],
        "marker": {"dense": "D", "factorized": "o", "mlp": "^"}[architecture],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["dflash", "eagle3"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry_path = Path(
        f"reports/mapper-scaling-20260905/target14b-{args.family}-fit-results.json"
    )
    registry = json.loads(registry_path.read_text())
    if (
        registry.get("status") != "complete"
        or registry["family"] != args.family
        or len(registry["results"]) != 16
    ):
        raise ValueError(
            "fitting visualization requires the full audited family matrix"
        )
    for name, sha in registry["input_sha256"].items():
        if digest(Path(name)) != sha:
            raise ValueError("audited fitting input changed")
    results = sorted(
        registry["results"].values(),
        key=lambda r: (
            r["trial"]["distinct_examples"],
            {"dense": 0, "factorized": 1, "mlp": 2}[r["trial"]["architecture"]],
            r["trial"].get("width", 0),
            r["trial"]["seed"],
        ),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    # The fixed 11.6-inch canvas is reduced to about 47% in the manuscript.
    # Source fonts of 17--18 pt remain readable at roughly 8--8.5 pt there.
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True)
    for ax, n in zip(axes, [512, 2048], strict=True):
        for record in results:
            trial = record["trial"]
            if trial["distinct_examples"] != n:
                continue
            points = [p for p in record["validation_trajectory"] if p["step"] > 0]
            appearance = style(trial)
            display = label(trial)
            if trial["architecture"] == "factorized":
                display = f"Linear {trial['width']}"
            if trial["seed"] != 1729:
                appearance.update(color="#777777", linestyle="--", marker="x")
                display += ", seed 1730"
            ax.plot(
                [p["step"] for p in points],
                [p["groups"]["validation"]["objective"] for p in points],
                label=display,
                linewidth=2.2,
                markersize=6,
                markeredgewidth=1.1,
                markerfacecolor="white"
                if trial["architecture"] == "factorized"
                else appearance["color"],
                **appearance,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([128, 512, 2048, 8192], ["128", "512", "2,048", "8,192"])
        ax.set_xlabel("Updates (batch size 4)")
        ax.set_title(f"{n:,} training examples")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Validation relative MSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=17,
        handlelength=1.8, handletextpad=0.45, columnspacing=1.0, labelspacing=0.4,
    )
    family_label = {"dflash": "DFlash", "eagle3": "EAGLE-3"}[args.family]
    fig.suptitle(f"{family_label} / Qwen3-14B: fixed-data fitting trajectories")
    fig.tight_layout(rect=(0, 0.21, 1, 0.95), pad=0.6, w_pad=0.8)
    for extension in ["png", "pdf"]:
        fig.savefig(
            args.output / f"validation-trajectories.{extension}",
            dpi=160,
            metadata={"CreationDate": None, "ModDate": None}
            if extension == "pdf"
            else {"Software": "RelaySpec"},
        )
    plt.close(fig)
    lines = [
        f"# {family_label} 14B fitting diagnostics",
        "",
        "All 16 fits passed the raw-log audit. Training uses 512 or 2,048 distinct examples, 8,192 batch-four updates and the same 1,024-record validation set. No larger-data scaling is resumed. These are feature-fitting measurements, not decoding rankings or answer-quality claims.",
        "",
        "![Validation trajectories](validation-trajectories.png)",
        "",
        "Step 0 is omitted from the plot for readability and retained in the registry. Both dense fitting seeds are shown separately. Feature-validation minima do not change the fixed 8,192-update decoding endpoint.",
        "",
        "| N | Mapper | Seed | Parameters (M) | Train loss | Validation loss | Best saved step | Update seconds | Total worker seconds |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        t = r["trial"]
        lines.append(
            f"| {t['distinct_examples']} | {label(t)} | {t['seed']} | {r['parameters'] / 1e6:.3f} | {r['train_objective']:.5f} | {r['validation_objective']:.5f} | {r['best_validation_step']} | {r['timing']['training_update_seconds']:.2f} | {r['timing']['worker_total_seconds']:.2f} |"
        )
    lines += [
        "",
        "Train loss uses the declared 256-record diagnostic prefix; validation uses all 1,024 records. Worker time includes setup, initial and checkpoint validation, input preparation, fitting and export. Four independent fits share each allocation; sum of worker times is not batch wall time.",
        "",
        "Widths 512/1,024/4,096 have 14.418/28.836/115.343 million parameters in both factorized linear and MLP maps; dense has 65.536 million. A linear width above output dimension 2,560 does not increase its maximum function-class rank.",
        "",
        "## Late validation changes",
        "",
        "| N | Mapper | Seed | Minimum saved loss | Endpoint loss | Endpoint change from minimum |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in results:
        t = r["trial"]
        minimum = min(
            p["groups"]["validation"]["objective"] for p in r["validation_trajectory"]
        )
        lines.append(
            f"| {t['distinct_examples']} | {label(t)} | {t['seed']} | {minimum:.5f} | {r['validation_objective']:.5f} | {100 * (r['validation_objective'] / minimum - 1):+.2f}% |"
        )
    lines += [
        "",
        "These descriptive minima come from exposed feature validation. Small increases can reflect optimization fluctuations; comparisons of selected minima do not establish a decoding improvement. Wide-model learning-rate checks and paired endpoint decoding remain separate.",
        "",
    ]
    report_path = args.output / "FITTING.md"
    report_path.write_text("\n".join(lines))
    (args.output / "evidence.json").write_text(
        json.dumps(
            {
                "status": "fitting_only",
                "family": args.family,
                "registry": str(registry_path),
                "registry_sha256": digest(registry_path),
                "builder_sha256": digest(Path(__file__)),
                "artifacts": {
                    p.name: digest(p)
                    for p in [
                        report_path,
                        args.output / "validation-trajectories.png",
                        args.output / "validation-trajectories.pdf",
                    ]
                },
                "scope": "Complete fitting trajectories and costs only; decoding/quality remain separately gated.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
