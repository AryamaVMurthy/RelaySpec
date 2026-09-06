"""Declare EAGLE small-data answer-quality evaluation from completed fitting evidence."""

import argparse
import copy
import json
import random
from pathlib import Path

import yaml
from summarize_focused_scaling import collect, digest


def manifest_for_order(records, seed):
    """Benchmark selection shuffles its input; invert that permutation for shards."""
    permutation = list(range(len(records)))
    random.Random(seed).shuffle(permutation)
    result = [None] * len(records)
    for index, position in enumerate(permutation):
        result[position] = records[index]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = Path("reports/mapper-scaling-20260905")
    matrix_path = Path("configs/submission/scaling/matrix-eagle3-small-v1/matrix.json")
    fits_path = base / "eagle3-capacity-results.json"
    source_config = Path(
        "configs/submission/scaling/campaign-eagle3-capacity-complete.yaml"
    )
    source_provenance = source_config.with_suffix(".provenance.json")
    fits = collect(args.raw_root, matrix_path)
    if fits != json.loads(fits_path.read_text()) or fits["status"] != "complete":
        raise ValueError(
            "EAGLE quality requires the complete source-verified fitting matrix"
        )
    config = yaml.safe_load(source_config.read_text())
    provenance = json.loads(source_provenance.read_text())
    if (
        digest(source_config) != provenance["config_sha256"]
        or digest(matrix_path) != provenance["matrix_sha256"]
    ):
        raise ValueError("EAGLE endpoint campaign provenance changed")
    if (
        config["proposer"]["family"] != "eagle3"
        or config["target"]["id"] != "Qwen/Qwen3-8B"
    ):
        raise ValueError("this declaration is for the pinned EAGLE-3 8B interface")
    selected = [
        f"relay_eagle3_{kind}_n{n}"
        for kind in ["dense", "factorized2048", "mlp2048"]
        for n in [512, 2048]
    ] + ["relay_eagle3_factorized512_n2048", "relay_eagle3_mlp512_n2048"]
    inputs = {
        str(p): digest(p)
        for p in [matrix_path, fits_path, source_config, source_provenance]
    }
    variants = {}
    for alias in selected:
        declaration = provenance["variants"][alias]
        trial = declaration["trial"]
        fit = fits["results"][trial["name"]]
        folder = args.raw_root / fit["raw_directory"]
        gate_path = folder.parent.parent / "batch-gate.json"
        gate = json.loads(gate_path.read_text())
        path = config["relay_probe"]["variants"][alias]
        if (
            trial != fit["trial"]
            or trial["steps"] != 8192
            or trial["distinct_examples"] not in {512, 2048}
            or trial["normalize_input"] is not False
            or declaration["fit_gate_sha256"] != digest(gate_path)
            or declaration["checkpoint_sha256"] != gate["checkpoint_sha256"].get(path)
            or Path(path).name != "step-008192.pt"
        ):
            raise ValueError("selected EAGLE checkpoint differs from audited endpoint")
        for name in ["fit-complete.json", "validation.jsonl"]:
            inputs[str(folder / name)] = digest(folder / name)
        inputs[str(gate_path)] = digest(gate_path)
        variants[alias] = {**declaration, "checkpoint": path}
    manifest_path = Path(config["benchmark"]["manifest_path"])
    inputs[str(manifest_path)] = digest(manifest_path)
    records = [
        r
        for r in json.loads(manifest_path.read_text())["records"]
        if r["benchmark"] == "math500"
    ]
    random.Random(config["seed"]).shuffle(records)
    records = records[:128]
    if len(records) != 128 or len({r["problem_id"] for r in records}) != 128:
        raise ValueError("quality needs 128 unique exposed MATH requests")
    config["generation"]["max_new_tokens"] = 2048
    config["relay_probe"]["variants"] = {
        alias: variants[alias]["checkpoint"] for alias in selected
    }
    config["benchmark"]["methods"] = [
        "native_ar",
        "native_target_eagle3",
        "source_reuse_eagle3",
        *selected,
    ]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stages = {}
    for stage, chosen in [
        ("pilot", records[:8]),
        ("full0", records[:64]),
        ("full1", records[64:]),
    ]:
        subset_path = args.output_dir / f"manifest-{stage}.json"
        subset_path.write_text(
            json.dumps(
                {
                    "records": manifest_for_order(chosen, config["seed"]),
                    "scope": "Exposed MATH development subset. No untouched confirmation data.",
                },
                indent=2,
            )
            + "\n"
        )
        stage_config = copy.deepcopy(config)
        stage_config["run_name"] = f"eagle3-small-data-quality-{stage}"
        stage_config["benchmark"].update(
            manifest_path=str(subset_path), max_prompts=len(chosen)
        )
        path = args.output_dir / f"campaign-{stage}.yaml"
        path.write_text(yaml.safe_dump(stage_config, sort_keys=False))
        stages[stage] = {
            "config": str(path),
            "config_sha256": digest(path),
            "manifest": str(subset_path),
            "manifest_sha256": digest(subset_path),
            "requests": len(chosen),
            "problem_ids": [r["problem_id"] for r in chosen],
        }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "status": "declared_before_long_output_eagle_evaluation",
                "large_data_scaling": "paused",
                "input_sha256": inputs,
                "target": config["target"],
                "proposer": config["proposer"],
                "feature_cache_index_sha256": fits["feature_cache_index_sha256"],
                "methods": config["benchmark"]["methods"],
                "variants": variants,
                "stages": stages,
                "dense_reference": "relay_eagle3_dense_n2048",
                "output_cap": 2048,
                "quality_margin": 0.01,
                "quality_interval": "Bonferroni combination of exact Clopper-Pearson discordance intervals; pinned scipy1.16.1, IID request-pair assumption, conservative95% coverage. Report inconclusive outcomes without changing the margin.",
                "selection": "Eight existing8192-update endpoints: dense/factor2048/MLP2048 at both512/2048 examples and factor512/MLP512 at2048. Structural anchors selected after exposed short-output capacity decoding, before this quality evaluation. Width128 remains in the complete short-output grid but is not a quality finalist. No outcome-dependent horizon change.",
                "execution": "Pilot8requests x11methods,2048-token cap,540-second process and10-minute Slurm limits. Only after its completeness/scoring/source/resource audit passes, run the two64-request disjoint full shards. Preserve original128-request membership/order within shards; method rotation restarts per shard. Four GPUs total, sequential allocations. Report source, native and mapper costs under this exact EAGLE runtime.",
                "scope": "Exposed development answer-quality evaluation. Compare all selected mappers with matched AR and dense2048, including cap-length counts and conservative paired quality intervals. Correctness at this cap does not establish uncapped quality; cross-runtime absolute TPS is not an algorithm-only ranking. No training or distinct-data expansion, and no untouched-confirmation claim.",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Declared {len(selected)} EAGLE mappers, three controls,8-request pilot and128-request development comparison"
    )


if __name__ == "__main__":
    main()
