"""Audit all frozen sampling waves and bootstrap questions across both seeds."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml
from aggregate_results import build_math_scorer

from relayspec.sampling_rng import initialize_sampling_rng


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clustered_summary(rows, reference, seeds):
    methods = sorted({r["method"] for r in rows})
    ids = sorted({r["problem_id"] for r in rows})
    lookup = {(r["problem_id"], r["study_seed"], r["method"]): r for r in rows}
    expected = {(p, s, m) for p in ids for s in seeds for m in methods}
    if len(lookup) != len(rows) or set(lookup) != expected:
        raise ValueError("Incomplete or duplicated question/seed/method cells")
    arrays = {}
    for m in methods:
        arrays[m] = np.array(
            [
                [
                    sum(lookup[p, s, m][key] for s in seeds)
                    for key in ("output_tokens", "request_seconds", "correct")
                ]
                for p in ids
            ],
            dtype=float,
        )
        if not np.isfinite(arrays[m]).all() or (arrays[m][:, :2] <= 0).any():
            raise ValueError("Invalid timing or token totals")
    indices = np.random.default_rng(1729).integers(0, len(ids), (10000, len(ids)))
    draws = {m: a[indices].sum(axis=1) for m, a in arrays.items()}
    ref = arrays[reference].sum(axis=0)
    result = {
        "questions": len(ids),
        "outputs_per_method": len(ids) * len(seeds),
        "reference": reference,
        "methods": {},
    }
    n = result["outputs_per_method"]
    for m, a in arrays.items():
        tokens, seconds, correct = a.sum(axis=0)
        sampled = draws[m]
        ratios = (sampled[:, 0] / sampled[:, 1]) / (
            draws[reference][:, 0] / draws[reference][:, 1]
        )
        result["methods"][m] = {
            "tokens_per_second": tokens / seconds,
            "throughput_ratio": (tokens / seconds) / (ref[0] / ref[1]),
            "throughput_ci95": np.quantile(ratios, [0.025, 0.975]).tolist(),
            "request_time_ratio": ref[1] / seconds,
            "request_time_ci95": np.quantile(
                draws[reference][:, 1] / sampled[:, 1], [0.025, 0.975]
            ).tolist(),
            "mean_request_seconds": seconds / n,
            "mean_output_tokens": tokens / n,
            "correct": int(correct),
            "accuracy": correct / n,
            "accuracy_difference": (correct - ref[2]) / n,
            "accuracy_difference_ci95": np.quantile(
                (sampled[:, 2] - draws[reference][:, 2]) / n, [0.025, 0.975]
            ).tolist(),
            "cap_count": sum(
                r["output_tokens"] == 2048 for r in rows if r["method"] == m
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring-repo", type=Path, required=True)
    args = parser.parse_args()
    repo = Path.cwd()
    root = repo / "reports/autoresearch-20260907"
    base = repo / "configs/autoresearch/20260907/sampling-breadth-v1"
    study = repo / "configs/autoresearch/20260907/dflash-sampling-matched-v1"
    protocol = json.loads((study / "protocol.json").read_text())
    inherited = json.loads((base / "protocol.json").read_text())
    ledger = json.loads((root / "jobs.json").read_text())["jobs"]
    assert digest(base / "manifest32.json") == inherited["manifest_sha256"]
    frozen_ids = {
        r["problem_id"]
        for r in json.loads((base / "manifest32.json").read_text())["records"]
    }
    assert len(frozen_ids) == 32
    inputs = {
        str((base / "manifest32.json").relative_to(repo)): digest(
            base / "manifest32.json"
        )
    }
    cells = {}
    for declaration in protocol["waves"]:
        number = declaration["wave"]
        entries = [j for j in ledger if j["wave"] == number]
        assert len(entries) == 1
        entry = entries[0]
        run = root / f"run-{entry['id']}"
        source = run / "source-commit.txt"
        assert source.read_text().strip() == entry["source_commit"]
        status = run / "wave-result.json"
        lanes = json.loads(status.read_text())["lanes"]
        assert len(lanes) == 4 and all(r["status"] == "pass" for r in lanes)
        wave = repo / f"configs/autoresearch/20260907/wave{number}.json"
        for p in [source, status, wave]:
            inputs[str(p.relative_to(repo))] = digest(p)
        for i, spec in enumerate(json.loads(wave.read_text())["lanes"]):
            lane = run / f"lane{i}"
            actual_spec = run / f"lane{i}-spec.json"
            assert json.loads(actual_spec.read_text()) == spec
            config_path = repo / spec["config"]
            config = yaml.safe_load(config_path.read_text())
            actual = yaml.safe_load((lane / "config.yaml").read_text())
            for field in ["generation", "target", "proposer", "native_target_proposer"]:
                assert actual[field] == config[field]
            seed = config["benchmark"]["sampling_seed_base"]
            assert (
                seed
                == declaration["seed_base"]
                == actual["benchmark"]["sampling_seed_base"]
            )
            manifest = repo / config["benchmark"]["manifest_path"]
            assert manifest == base / f"shard{declaration['shards'][i // 2]}.json"
            ids = {r["problem_id"] for r in json.loads(manifest.read_text())["records"]}
            assert len(ids) == 8 and spec["max_new_tokens"] == 2048
            native = (
                "native_target_dflash"
                if spec["family"] == "dflash"
                else "native_target_eagle3"
            )
            methods = {"native_ar", "relay_base", native}
            assert set(actual["benchmark"]["methods"]) == methods
            path = lane / "benchmark-rank0.jsonl"
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            assert len(rows) == 24 and {
                (r["problem_id"], r["method"], r["repetition"]) for r in rows
            } == {(p, m, 0) for p in ids for m in methods}
            temp = config["generation"]["temperature"]
            for r in rows:
                assert r["sampling_temperature"] == temp
                assert r["sampling_seed"] == initialize_sampling_rng(
                    temp, seed, r["problem_id"], 0, r["turn_index"]
                )
                if r["method"] == "relay_base":
                    assert (
                        r["mapper_checkpoint_sha256"]
                        == inherited["checkpoints"][spec["checkpoint"]]
                    )
                r.update(
                    study_seed=seed, study_family=spec["family"], study_temperature=temp
                )
            cells.setdefault((spec["family"], temp), []).extend(rows)
            for p in [
                actual_spec,
                config_path,
                manifest,
                path,
                lane / "config.yaml",
                lane / "completion-gate.json",
            ]:
                inputs[str(p.relative_to(repo))] = digest(p)
    assert len(cells) == 2 and sum(map(len, cells.values())) == 384
    os.chdir(args.scoring_repo)
    provenance_path = Path("reports/ar-revision-20260905/scorer-provenance.json")
    provenance = json.loads(provenance_path.read_text())
    assert all(digest(Path(p)) == sha for p, sha in provenance["sha256"].items())
    scorer, name = build_math_scorer(Path("vendor/qwen-math/evaluation"))
    cache = {}
    results = []
    for (family, temp), rows in sorted(cells.items()):
        assert {r["problem_id"] for r in rows} == frozen_ids
        for r in rows:
            key = (r["completion"], str(r["reference_answer"]), r["benchmark"])
            if key not in cache:
                cache[key] = scorer(*key)
            r.update(cache[key])
        native = (
            "native_target_dflash" if family == "dflash" else "native_target_eagle3"
        )
        results.append(
            dict(
                family=family,
                temperature=temp,
                ar_reference=clustered_summary(
                    rows, "native_ar", protocol["seeds"]
                ),
                native_reference=clustered_summary(
                    rows, native, protocol["seeds"]
                ),
            )
        )
    output = dict(
        status="complete",
        results=results,
        input_sha256=inputs,
        protocol_sha256=digest(study / "protocol.json"),
        scorer=name,
        scorer_provenance_sha256=digest(provenance_path),
        scope=protocol["analysis"],
        exposure=protocol["exposure"],
    )
    (root / "dflash-sampling-matched-summary.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    (root / "dflash-sampling-matched-scored.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for rows in cells.values() for r in rows)
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
