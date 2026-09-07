"""Audit frozen sampling cells and score all saved full-answer outputs."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import yaml
from aggregate_results import build_math_scorer

from relayspec.ar_paper_evidence import summarize
from relayspec.paired_accuracy import paired_accuracy_interval
from relayspec.sampling_rng import initialize_sampling_rng


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=int, default=28559)
    parser.add_argument("--wave", type=int, default=64)
    parser.add_argument("--scoring-repo", type=Path, required=True)
    args = parser.parse_args()
    repo = Path.cwd()
    root = repo / "reports/autoresearch-20260907"
    run = root / f"run-{args.job}"
    wave = repo / f"configs/autoresearch/20260907/wave{args.wave}.json"
    specs = json.loads(wave.read_text())["lanes"]
    ledger = json.loads((root / "jobs.json").read_text())["jobs"]
    entry = next(x for x in ledger if x["id"] == args.job)
    assert (run / "source-commit.txt").read_text().strip() == entry["source_commit"]
    status = json.loads((run / "wave-result.json").read_text())["lanes"]
    assert len(status) == 4 and all(x["status"] == "pass" for x in status)
    inputs = {str(wave.relative_to(repo)): digest(wave)}
    cells = []
    common_ids = None
    for i, spec in enumerate(specs):
        lane = run / f"lane{i}"
        assert json.loads((run / f"lane{i}-spec.json").read_text()) == spec
        expected = yaml.safe_load((repo / spec["config"]).read_text())
        actual = yaml.safe_load((lane / "config.yaml").read_text())
        assert actual["generation"] == expected["generation"]
        assert (
            actual["benchmark"]["sampling_seed_base"]
            == expected["benchmark"]["sampling_seed_base"]
        )
        assert (
            actual["proposer"] == expected["proposer"]
            and actual["target"] == expected["target"]
        )
        methods = {
            "native_ar",
            "relay_base",
            "native_target_dflash"
            if spec["family"] == "dflash"
            else "native_target_eagle3",
        }
        assert set(actual["benchmark"]["methods"]) == methods
        manifest = repo / expected["benchmark"]["manifest_path"]
        selected = [
            r
            for r in json.loads(manifest.read_text())["records"]
            if r["benchmark"] == "gsm8k"
        ]
        random.Random(expected["seed"]).shuffle(selected)
        ids = {r["problem_id"] for r in selected[:8]}
        if common_ids is None:
            common_ids = ids
        assert ids == common_ids and len(ids) == 8
        path = lane / "benchmark-rank0.jsonl"
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        assert len(rows) == 24 and {
            (r["problem_id"], r["method"], r["repetition"]) for r in rows
        } == {(p, m, 0) for p in ids for m in methods}
        for r in rows:
            temp = expected["generation"]["temperature"]
            assert r["sampling_temperature"] == temp
            assert r["sampling_seed"] == initialize_sampling_rng(
                temp,
                expected["benchmark"]["sampling_seed_base"],
                r["problem_id"],
                r["repetition"],
                r["turn_index"],
            )
            if r["method"] == "relay_base":
                assert (
                    r["mapper_checkpoint_sha256"]
                    == spec["checkpoint_sha256"][spec["checkpoint"]]
                )
        for p in [
            path,
            manifest,
            lane / "config.yaml",
            run / f"lane{i}-spec.json",
            run / f"lane{i}-status.json",
            lane / "completion-gate.json",
        ]:
            inputs[str(p.relative_to(repo))] = digest(p)
        cells.append(
            dict(
                family=spec["family"],
                temperature=expected["generation"]["temperature"],
                sampling_seed_base=expected["benchmark"]["sampling_seed_base"],
                rows=rows,
            )
        )
    os.chdir(args.scoring_repo)
    provenance_path = Path("reports/ar-revision-20260905/scorer-provenance.json")
    provenance = json.loads(provenance_path.read_text())
    assert all(digest(Path(p)) == sha for p, sha in provenance["sha256"].items())
    scorer, name = build_math_scorer(Path("vendor/qwen-math/evaluation"))
    cache = {}
    scored = []
    for cell in cells:
        rows = cell.pop("rows")
        for r in rows:
            key = (r["completion"], str(r["reference_answer"]), r["benchmark"])
            if key not in cache:
                cache[key] = scorer(*key)
            r.update(cache[key])
        cell["ar_reference"] = summarize(rows, reference="native_ar")
        native = (
            "native_target_dflash"
            if cell["family"] == "dflash"
            else "native_target_eagle3"
        )
        cell["native_reference"] = summarize(rows, reference=native)
        lookup = {(r["problem_id"], r["method"]): r for r in rows}
        ids = sorted(common_ids)
        cell["paired_accuracy"] = paired_accuracy_interval(
            [lookup[p, "relay_base"]["correct"] for p in ids],
            [lookup[p, "native_ar"]["correct"] for p in ids],
        )
        cell["cap_counts"] = {
            m: sum(r["output_tokens"] == 2048 for r in rows if r["method"] == m)
            for m in cell["ar_reference"]["methods"]
        }
        scored.extend(
            dict(
                **r, study_family=cell["family"], study_temperature=cell["temperature"]
            )
            for r in rows
        )
    result = dict(
        results=cells,
        input_sha256=inputs,
        scorer=name,
        scorer_provenance_sha256=digest(provenance_path),
        scope="Four descriptive cells reuse the same eight exposed questions. One explicit per-request sampling seed per cell.2048token cap. Different sampled outputs and lengths are expected. Paired request intervals condition on this seed, do not establish quality noninferiority or distributional exactness, and do not represent32 independent questions.",
    )
    (root / "sampling-full-answer-summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (run / "sampling-scored.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in scored)
    )
    for cell in cells:
        print(
            cell["family"],
            cell["temperature"],
            {
                m: dict(
                    tps=round(v["tokens_per_second"], 2),
                    ar_ratio=round(v["throughput_ratio"], 3),
                    correct=round(8 * v["accuracy"]),
                )
                for m, v in cell["ar_reference"]["methods"].items()
            },
            cell["cap_counts"],
        )


if __name__ == "__main__":
    main()
