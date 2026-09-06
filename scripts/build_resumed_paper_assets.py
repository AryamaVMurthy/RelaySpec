"""Export paper tables and a domain-composition plot from audited resumed results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from relayspec.cached_fit_evidence import digest


def checked(path):
    data = json.loads(path.read_text())
    assert data["status"] == "complete"
    for source, sha in data["input_sha256"].items():
        assert digest(Path(source)) == sha, source
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("paper/iclr2027"))
    parser.add_argument("--fitting-and-confirmation-only", action="store_true")
    args = parser.parse_args()
    root = args.raw_root / "reports/mapper-scaling-20260905"
    output = args.output / "generated"
    figures = args.output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    validation_proof = json.loads(
        (root / "resumed-composition/common-validation-identity.json").read_text()
    )
    validation_entries = []
    for path, sha in validation_proof["input_sha256"].items():
        assert digest(Path(path)) == sha
        validation_entries.append(
            [
                e
                for e in json.loads(Path(path).read_text())["entries"]
                if e["split"] == "validation"
            ]
        )
    assert len(validation_entries) == 2
    assert len(validation_entries[0]) == 2048
    assert validation_entries[0] == validation_entries[1]
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Family & Examples & Tokens/s & Retention (95\% CI) & Correct & At cap \\",
        r"\midrule",
    ]
    summary = {}
    controls = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Family & Method & Tokens/s & Correct & At cap \\",
        r"\midrule",
    ]
    for family, label in [("dflash", "DFlash"), ("eagle3", "EAGLE-3")]:
        path = root / "resumed-confirmation" / f"{family}-full-audit.json"
        data = checked(path)
        protocol = json.loads(
            Path(
                f"configs/submission/confirmation-gsm8k-20260906/frozen-dense-v1/{family}-protocol.json"
            ).read_text()
        )
        candidate, reference = protocol["candidate"], protocol["dense_reference"]
        metrics = data["comparisons"][reference]["paired_throughput"]["methods"]
        accuracy = data["comparisons"]["native_ar"]["conservative_paired_accuracy"][
            candidate
        ]
        for method in protocol["methods"]:
            method_label = (
                "AR"
                if method == "native_ar"
                else "Native drafter"
                if method.startswith("native_target")
                else "Source reuse"
                if "source" in method
                else "Dense 512"
                if method == candidate
                else "Dense 2,048"
            )
            controls.append(
                f"{label} & {method_label} & {metrics[method]['tokens_per_second']:.1f} & "
                f"{data['methods'][method]['correct_count']}/256 & "
                f"{data['methods'][method]['cap_length_outputs']} " + r"\\"
            )
        summary[family] = {
            "candidate": candidate,
            "reference": reference,
            "candidate_metrics": metrics[candidate],
            "paired_accuracy_against_ar": accuracy,
            "speed_criterion_met": metrics[candidate]["throughput_ci95"][0]
            > protocol["speed_threshold"],
            "quality_criterion_met": accuracy["confidence_interval"][0]
            > -protocol["quality_margin"],
        }
        for method, n in [(candidate, 512), (reference, 2048)]:
            m = metrics[method]
            lo, hi = m["throughput_ci95"]
            ret = (
                f"{100 * m['throughput_ratio']:.1f} [{100 * lo:.1f},{100 * hi:.1f}]"
                if n == 512
                else "100 (reference)"
            )
            lines.append(
                f"{label} & {n:,} & {m['tokens_per_second']:.1f} & {ret} & {data['methods'][method]['correct_count']}/256 & {data['methods'][method]['cap_length_outputs']} "
                + r"\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "frozen_confirmation_table.tex").write_text("\n".join(lines) + "\n")
    controls.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "frozen_confirmation_controls_table.tex").write_text(
        "\n".join(controls) + "\n"
    )
    fit_lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Mapper & Training & Train loss & Math val. & General val. & Fit seconds \\",
        r"\midrule",
    ]
    for arm in ["math", "mixed"]:
        data = checked(root / "resumed-composition" / f"{arm}-fits-audit.json")
        for name, result in data["results"].items():
            trial = result["trial"]
            kind = trial["architecture"]
            label = {"dense": "Dense", "factorized": "Linear-512", "mlp": "MLP-512"}[
                kind
            ]
            if trial["seed"] == 1730:
                label += " (seed 2)"
            groups = result["validation_trajectory"][-1]["groups"]["validation"][
                "by_domain"
            ]
            fit_lines.append(
                f"{label} & {arm} & {result['train_objective']:.4f} & {groups['math']['objective']:.4f} & {groups['general_instruction']['objective']:.4f} & {result['timing']['training_update_seconds']:.1f} "
                + r"\\"
            )
    fit_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "composition_fitting_table.tex").write_text("\n".join(fit_lines) + "\n")
    if args.fitting_and_confirmation_only:
        print(
            "Exported completed fitting and confirmation tables; decoding is separate"
        )
        return
    decode = checked(root / "resumed-composition-evaluation/full-audit.json")
    tasks = list(decode["tasks"])
    labels = {
        "math500": "MATH",
        "gsm8k": "GSM8K",
        "humaneval": "HumanEval",
        "mtbench": "MT-Bench",
    }
    methods = list(decode["tasks"][tasks[0]]["composition_pairs"])
    names = ["Dense", "Linear-512", "MLP-512", "Dense (seed 2)"]
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Task & Mapper & Math tokens/s & Mixed tokens/s & Ratio (95\% CI) \\",
        r"\midrule",
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for index, (method, label) in enumerate(zip(methods, names, strict=True)):
        x = np.arange(len(tasks)) + (index - 1.5) * 0.14
        ys = []
        los = []
        his = []
        for task in tasks:
            pair = decode["tasks"][task]["composition_pairs"][method]
            mixed = pair["methods"][pair["mixed_method"]]
            math = pair["methods"][method]
            lo, hi = mixed["throughput_ci95"]
            value = mixed["throughput_ratio"]
            ys.append(value)
            los.append(value - lo)
            his.append(hi - value)
            lines.append(
                f"{labels[task]} & {label} & {math['tokens_per_second']:.1f} & {mixed['tokens_per_second']:.1f} & {value:.3f} [{lo:.3f},{hi:.3f}] "
                + r"\\"
            )
        ax.errorbar(
            x,
            ys,
            yerr=[los, his],
            fmt=["o", "s", "^", "D"][index],
            capsize=3,
            label=label,
            markersize=4,
        )
    ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(np.arange(len(tasks)), [labels[t] for t in tasks])
    ax.set_ylabel("Mixed / math-only throughput")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    for suffix in ["pdf", "png"]:
        fig.savefig(
            figures / f"composition_throughput.{suffix}",
            dpi=220,
            bbox_inches="tight",
            metadata={"CreationDate": None, "ModDate": None}
            if suffix == "pdf"
            else None,
        )
    plt.close(fig)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "composition_decoding_table.tex").write_text("\n".join(lines) + "\n")
    (output / "resumed_paper_findings.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (output / "resumed_evidence_registry.json").write_text(
        json.dumps(
            {"raw_root": str(args.raw_root.resolve()), "status": "complete"}, indent=2
        )
        + "\n"
    )
    print("Exported confirmation, composition fitting and decoding assets")


if __name__ == "__main__":
    main()
