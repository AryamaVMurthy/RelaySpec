"""Bind common decoding to the audited, predeclared warm-budget checkpoints."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from relayspec.ar_paper_evidence import read_rows, summarize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits", type=Path, nargs=3, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        audited = folder / "fits.json"
        campaign = folder / "campaign.yaml"
        for script, output in [
            ("audit_timed_budgets.py", audited),
            ("build_timed_budget_campaign.py", campaign),
        ]:
            subprocess.run(
                [
                    sys.executable,
                    f"scripts/{script}",
                    "--runs",
                    *map(str, args.fits),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        fits = json.loads(audited.read_text())
        expected_config = campaign.read_bytes()
        expected_provenance = campaign.with_suffix(".provenance.json").read_bytes()
    config_path = args.run / "campaign-config.yaml"
    provenance_path = config_path.with_suffix(".provenance.json")
    if (
        fits["status"] != "complete"
        or config_path.read_bytes() != expected_config
        or provenance_path.read_bytes() != expected_provenance
    ):
        raise ValueError("decoding declaration differs from audited timed fits")
    config = yaml.safe_load(expected_config)
    provenance = json.loads(expected_provenance)
    gate_path, analysis_path = (
        args.run / "completion-gate.json",
        args.run / "analysis.json",
    )
    gate, analysis = (
        json.loads(gate_path.read_text()),
        json.loads(analysis_path.read_text()),
    )
    raw_files = sorted(args.run.glob("benchmark-rank*.jsonl"))
    if (
        gate["status"] != "pass"
        or analysis["status"] != "complete"
        or yaml.safe_load((args.run / "config.yaml").read_text()) != config
        or {p.name for p in raw_files} != set(analysis["raw_sha256"])
        or any(digest(p) != analysis["raw_sha256"][p.name] for p in raw_files)
    ):
        raise ValueError("raw timed decoding is incomplete or changed")
    rows = read_rows(args.run)
    if len(rows) != gate["records"] or {r["method"] for r in rows} != set(
        config["benchmark"]["methods"]
    ):
        raise ValueError("timed decoding omits declared methods")
    for row in rows:
        variant = provenance["variants"].get(row["method"])
        if variant and row.get("mapper_checkpoint_sha256") != variant["mapper_sha256"]:
            raise ValueError("timed decoding used another mapper")
        if (
            variant
            and "drafter_update_sha256" in variant
            and row.get("drafter_checkpoint_sha256") != variant["drafter_update_sha256"]
        ):
            raise ValueError("timed decoding used another drafter update")
        if row["output_tokens"] > config["generation"]["max_new_tokens"]:
            raise ValueError("timed decoding exceeded its declared token cap")
    against_initial = summarize(rows, reference="relay_initial")
    if against_initial["requests"] != gate["requests"] or gate["requests"] != 16:
        raise ValueError("timed decoding requires sixteen paired requests")
    matched = {}
    for multiplier in (0.25, 1, 4):
        prefix = f"relay_budget{round(multiplier * 100):03d}_"
        selected = [r for r in rows if r["method"].startswith(prefix)]
        matched[str(multiplier)] = summarize(selected, reference=prefix + "feature")
    curves = {}
    for objective in ("feature", "connector_ce", "lora32_s1729", "lora32_s1730"):
        methods = {
            f"relay_budget{round(m * 100):03d}_{objective}" for m in (0.25, 1, 4)
        }
        curves[objective] = summarize(
            [r for r in rows if r["method"] in methods],
            reference=f"relay_budget025_{objective}",
        )
    inputs = dict(fits["input_sha256"])
    for path in [
        config_path,
        provenance_path,
        gate_path,
        analysis_path,
        args.run / "config.yaml",
        *raw_files,
    ]:
        inputs[str(path)] = digest(path)
    result = {
        "status": "complete",
        "input_sha256": inputs,
        "fitting": fits["records"],
        "against_initial": against_initial,
        "against_dense8192_reference": summarize(
            rows, reference="relay_dense8192_reference"
        ),
        "matched_budget_against_feature": matched,
        "within_objective_against_quarter_budget": curves,
        "scope": "Twelve predeclared 512-example fits at three warm-training budgets. "
        "Initial mapper training is charged to CE and LoRA. Setup and cache construction "
        "are separate from warm training. Sixteen exposed development requests at a "
        "256-token cap. Paired request bootstrap intervals are descriptive, not fitting-seed "
        "uncertainty or multiple-comparison corrections. No full-answer quality or untouched "
        "confirmation claim. The two LoRA seeds vary the update initialization, not the "
        "shared initial mapper. No large-data expansion.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {
                    k: value[k]
                    for k in [
                        "tokens_per_second",
                        "throughput_ratio",
                        "throughput_ci95",
                    ]
                }
                for name, value in against_initial["methods"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
