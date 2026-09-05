"""Audit declared checkpoint campaigns and compare early stops with endpoints."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import yaml

from relayspec.ar_paper_evidence import read_rows, summarize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance_path = args.config.with_suffix(".provenance.json")
    provenance = json.loads(provenance_path.read_text())
    config = yaml.safe_load(args.config.read_text())
    gate_path = args.run / "completion-gate.json"
    gate = json.loads(gate_path.read_text())
    analysis_path = args.run / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    raw_files = sorted(args.run.glob("benchmark-rank*.jsonl"))
    if (
        digest(args.config) != provenance["config_sha256"]
        or yaml.safe_load((args.run / "config.yaml").read_text()) != config
        or gate["status"] != "pass"
        or analysis["status"] != "complete"
        or {p.name for p in raw_files} != set(analysis["raw_sha256"])
        or any(digest(p) != analysis["raw_sha256"][p.name] for p in raw_files)
    ):
        raise ValueError("campaign configuration or raw evidence changed")
    rows = read_rows(args.run)
    methods = set(config["benchmark"]["methods"])
    if len(rows) != gate["records"] or {r["method"] for r in rows} != methods:
        raise ValueError("campaign rows differ from the declared methods")
    variants = provenance["variants"]
    groups = defaultdict(dict)
    for name, variant in variants.items():
        if name not in methods:
            raise ValueError("declared checkpoint was not evaluated")
        step = variant.get("selected_step", variant.get("step"))
        if not isinstance(step, int) or step <= 0:
            raise ValueError("missing checkpoint step")
        trial = variant["trial"]["name"]
        if step in groups[trial]:
            raise ValueError("duplicate checkpoint for one fit")
        groups[trial][step] = name
    for row in rows:
        variant = variants.get(row["method"])
        if (
            variant
            and row.get("mapper_checkpoint_sha256") != variant["checkpoint_sha256"]
        ):
            raise ValueError("raw generation used another mapper checkpoint")
        if row["output_tokens"] > config["generation"]["max_new_tokens"]:
            raise ValueError("output exceeds declared token cap")
    against_reference = summarize(rows, reference=args.reference)
    if (
        against_reference["requests"] != gate["requests"]
        or gate["requests"] != config["benchmark"]["max_prompts"]
    ):
        raise ValueError("missing paired requests")
    comparisons = {}
    seed_groups = defaultdict(list)
    for trial, steps in groups.items():
        endpoint_steps = {variants[name]["trial"]["steps"] for name in steps.values()}
        if len(endpoint_steps) != 1:
            raise ValueError("inconsistent endpoint definition")
        endpoint_step = endpoint_steps.pop()
        if endpoint_step not in steps:
            raise ValueError("endpoint is missing")
        endpoint = steps[endpoint_step]
        settings = variants[endpoint]["trial"]
        setting = tuple(
            settings.get(k)
            for k in (
                "architecture",
                "width",
                "distinct_examples",
                "learning_rate",
                "l2_weight",
                "weight_decay",
                "normalize_input",
            )
        )
        seed_groups[setting].append((settings["seed"], steps, endpoint_step))
        for step, candidate in sorted(steps.items()):
            if step == endpoint_step:
                continue
            pair = summarize(
                [r for r in rows if r["method"] in {candidate, endpoint}],
                reference=endpoint,
            )
            comparisons[candidate] = {
                "trial": trial,
                "selected_step": step,
                "endpoint_step": endpoint_step,
                "reference": endpoint,
                **pair["methods"][candidate],
            }
    seed_panels = []
    for setting, fits in seed_groups.items():
        if len(fits) < 2:
            continue
        if len({seed for seed, _, _ in fits}) != len(fits):
            raise ValueError("duplicate fitting seed within one setting")
        panel = {
            "setting": dict(
                zip(
                    (
                        "architecture",
                        "width",
                        "distinct_examples",
                        "learning_rate",
                        "l2_weight",
                        "weight_decay",
                        "normalize_input",
                    ),
                    setting,
                    strict=True,
                )
            )
        }
        for selection in ("endpoint", "feature_validation"):
            records = []
            for seed, steps, endpoint_step in sorted(fits):
                endpoint = steps[endpoint_step]
                best = variants[endpoint].get("best_validation_step")
                if selection == "feature_validation" and best is None:
                    raise ValueError("seed panel requires declared validation minima")
                step = endpoint_step if selection == "endpoint" else best
                if step not in steps:
                    raise ValueError("seed panel is missing a selected checkpoint")
                method = steps[step]
                records.append(
                    {
                        "seed": seed,
                        "method": method,
                        "step": step,
                        **against_reference["methods"][method],
                    }
                )
            tps = [r["tokens_per_second"] for r in records]
            panel[selection] = {
                "seeds": records,
                "tokens_per_second_range": [min(tps), max(tps)],
            }
        seed_panels.append(panel)
    result = {
        "status": "complete",
        "run": str(args.run),
        "input_sha256": {
            str(p): digest(p)
            for p in [
                args.config,
                provenance_path,
                gate_path,
                analysis_path,
                *raw_files,
            ]
        },
        "variants": variants,
        "against_reference": against_reference,
        "early_vs_endpoint": comparisons,
        "seed_panels": seed_panels,
        "scope": "Paired development decoding at the declared token cap. "
        "Checkpoint choices were declared before decoding. Request bootstrap "
        "uses 10000 samples and seed 1729. These descriptive intervals do not "
        "adjust for multiple comparisons or establish task quality. Seed panels "
        "report observed throughput ranges separately from request intervals, "
        "not confidence intervals over fitting randomness. No training-data expansion.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name, values in comparisons.items():
        print(name, round(values["throughput_ratio"], 4), values["throughput_ci95"])


if __name__ == "__main__":
    main()
