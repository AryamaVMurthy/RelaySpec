"""Reproduce paired early-stop decoding evidence from source-verified campaigns."""

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
    inputs = {}
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Family & Mapper & $N$ & Learning rate & Selected update & Early / end [95\% CI] \\",
        r"\midrule",
    ]
    for family, name, run, config, reference in [
        (
            "DFlash",
            "convergence-checkpoint-decoding",
            "convergence-seed-wave/run-27817",
            "campaign-convergence-small",
            "relay_dense_n2048_step8192",
        ),
        (
            "EAGLE-3",
            "eagle3-checkpoint-decoding",
            "eagle-early-decoding/run-27826",
            "campaign-eagle3-capacity-validation",
            "relay_eagle3_dense_n2048_endpoint",
        ),
    ]:
        registry = Path("reports/mapper-scaling-20260905") / f"{name}.json"
        result = json.loads(registry.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            expected = Path(tmp) / "expected.json"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/summarize_checkpoint_decoding.py",
                    "--run",
                    str(args.raw_root / "reports/mapper-scaling-20260905" / run),
                    "--config",
                    f"configs/submission/scaling/{config}.yaml",
                    "--reference",
                    reference,
                    "--output",
                    str(expected),
                ],
                check=True,
                capture_output=True,
            )
            if json.loads(expected.read_text()) != result:
                raise ValueError("checkpoint evidence does not reproduce")
        inputs[str(registry)] = digest(registry)
        inputs.update(result["input_sha256"])
        for method, values in result["early_vs_endpoint"].items():
            trial = result["variants"][method]["trial"]
            label = {"dense": "Dense", "factorized": "Linear", "mlp": "MLP"}[
                trial["architecture"]
            ]
            if "width" in trial:
                label += f" {trial['width']}"
            lo, hi = values["throughput_ci95"]
            lines.append(
                f"{family} & {label} & {trial['distinct_examples']} & "
                f"{trial['learning_rate']:.4f} & {values['selected_step']} & "
                f"{values['throughput_ratio']:.3f} [{lo:.3f}, {hi:.3f}] " + r"\\"
            )
        if family == "DFlash":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    generated = args.output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "checkpoint_decoding_table.tex").write_text("\n".join(lines) + "\n")
    seed_registry = Path(
        "reports/mapper-scaling-20260905/selected-seed-decoding-results.json"
    )
    seeds = json.loads(seed_registry.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        expected = Path(tmp) / "expected.json"
        subprocess.run(
            [
                sys.executable,
                "scripts/summarize_checkpoint_decoding.py",
                "--run",
                str(
                    args.raw_root
                    / "reports/mapper-scaling-20260905/selected-seed-decoding/run-27836"
                ),
                "--config",
                "configs/submission/scaling/campaign-selected-seeds.yaml",
                "--reference",
                "relay_dense_n2048_s1729_step8192",
                "--output",
                str(expected),
            ],
            check=True,
            capture_output=True,
        )
        if json.loads(expected.read_text()) != seeds:
            raise ValueError("three-seed decoding does not reproduce")
    inputs[str(seed_registry)] = digest(seed_registry)
    inputs.update(seeds["input_sha256"])
    if len(seeds["seed_panels"]) != 6:
        raise ValueError("six declared seed panels are required")
    table = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Mapper & $N$ & Learning rate & Endpoint tok/s range & Selected tok/s range \\",
        r"\midrule",
    ]
    for panel in seeds["seed_panels"]:
        setting = panel["setting"]
        label = {"dense": "Dense", "factorized": "Linear", "mlp": "MLP"}[
            setting["architecture"]
        ]
        if setting["width"]:
            label += f" {setting['width']}"
        ranges = []
        for selection in ("endpoint", "feature_validation"):
            if {s["seed"] for s in panel[selection]["seeds"]} != {1729, 1730, 1731}:
                raise ValueError("seed panel is incomplete")
            lo, hi = panel[selection]["tokens_per_second_range"]
            ranges.append(f"{lo:.2f}--{hi:.2f}")
        table.append(
            f"{label} & {setting['distinct_examples']} & "
            f"{setting['learning_rate']:.4f} & " + " & ".join(ranges) + r" \\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "seed_decoding_table.tex").write_text("\n".join(table) + "\n")
    (generated / "checkpoint_evidence_registry.json").write_text(
        json.dumps(
            {
                "raw_root": str(args.raw_root.resolve()),
                "input_sha256": inputs,
                "scope": "All twelve distinct feature-validation minimum versus endpoint "
                "comparisons in the two declared development campaigns. Endpoints use "
                "8192 updates. Sixteen requests and 256-token cap in each campaign. "
                "No full-answer quality, fitting-seed uncertainty, or confirmation claim.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
