"""Audit paired prompt-style variants without treating them as new questions."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml
from aggregate_results import build_math_scorer

from relayspec.ar_paper_evidence import summarize
from relayspec.cached_fit_evidence import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring-repo", type=Path, required=True)
    args = parser.parse_args()
    repo = Path.cwd()
    root = repo / "reports/autoresearch-20260907"
    run = root / "run-28574"
    base = repo / "configs/autoresearch/20260907"
    wave = base / "wave79.json"
    protocol_path = base / "instruction-style-v1/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    job = next(
        j
        for j in json.loads((root / "jobs.json").read_text())["jobs"]
        if j["id"] == 28574
    )
    assert (run / "source-commit.txt").read_text().strip() == job["source_commit"]
    inputs = {
        str(p.relative_to(repo)): digest(p)
        for p in [wave, protocol_path, run / "source-commit.txt"]
    }
    cells = []
    for i, spec in enumerate(json.loads(wave.read_text())["lanes"]):
        lane = run / f"lane{i}"
        status = run / f"lane{i}-status.json"
        executed = run / f"lane{i}-spec.json"
        assert (
            json.loads(status.read_text())["status"] == "pass"
            and json.loads(executed.read_text()) == spec
        )
        config_path = repo / spec["config"]
        config = yaml.safe_load(config_path.read_text())
        actual = yaml.safe_load((lane / "config.yaml").read_text())
        for key in [
            "generation",
            "target",
            "proposer",
            "source_trunk",
            "native_target_proposer",
        ]:
            assert config[key] == actual[key]
        manifest = repo / config["benchmark"]["manifest_path"]
        records = {
            r["problem_id"]: r for r in json.loads(manifest.read_text())["records"]
        }
        native = (
            "native_target_dflash"
            if spec["family"] == "dflash"
            else "native_target_eagle3"
        )
        methods = {"native_ar", native, "relay_base"}
        assert set(actual["benchmark"]["methods"]) == methods
        path = lane / "benchmark-rank0.jsonl"
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        assert len(rows) == 48 and {
            (r["problem_id"], r["method"], r["repetition"]) for r in rows
        } == {(p, m, 0) for p in records for m in methods}
        styles = {r["instruction_style"] for r in records.values()}
        assert "original" in styles and len(styles) == 2
        for r in rows:
            record = records[r["problem_id"]]
            assert r["reference_answer"] == record["answer"]
            assert record["original_problem_id"] in protocol["base_problem_ids"]
            if r["method"] == "relay_base":
                assert (
                    r["mapper_checkpoint_sha256"]
                    == spec["checkpoint_sha256"][spec["checkpoint"]]
                )
            r.update(
                original_problem_id=record["original_problem_id"],
                instruction_style=record["instruction_style"],
                study_family=spec["family"],
            )
        cells.append(
            dict(
                family=spec["family"],
                style=next(s for s in styles if s != "original"),
                native=native,
                rows=rows,
            )
        )
        inputs.update(
            {
                str(p.relative_to(repo)): digest(p)
                for p in [
                    status,
                    executed,
                    config_path,
                    lane / "config.yaml",
                    manifest,
                    path,
                    lane / "completion-gate.json",
                ]
            }
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
        summaries = {}
        for r in rows:
            key = (r["completion"], str(r["reference_answer"]), r["benchmark"])
            if key not in cache:
                cache[key] = scorer(*key)
            r.update(cache[key])
        for style in ["original", cell["style"]]:
            selected = [r for r in rows if r["instruction_style"] == style]
            summaries[style] = dict(
                ar_reference=summarize(selected, reference="native_ar"),
                native_reference=summarize(selected, reference=cell["native"]),
                cap_counts={
                    m: sum(
                        r["output_tokens"] == 2048 for r in selected if r["method"] == m
                    )
                    for m in {r["method"] for r in selected}
                },
            )
        cell["conditions"] = summaries
        ids = sorted(protocol["base_problem_ids"])
        lookup = {
            (r["original_problem_id"], r["instruction_style"], r["method"]): r
            for r in rows
        }
        arrays = {}
        for style in ["original", cell["style"]]:
            for method in ["relay_base", cell["native"]]:
                arrays[style, method] = np.array(
                    [
                        [
                            lookup[p, style, method]["output_tokens"],
                            lookup[p, style, method]["request_seconds"],
                        ]
                        for p in ids
                    ]
                )
        indices = np.random.default_rng(1729).integers(0, 8, (10000, 8))
        retentions = {}
        for style in ["original", cell["style"]]:
            r, n = [
                arrays[style, m][indices].sum(axis=1)
                for m in ["relay_base", cell["native"]]
            ]
            retentions[style] = (r[:, 0] / r[:, 1]) / (n[:, 0] / n[:, 1])
        point = (
            summaries[cell["style"]]["native_reference"]["methods"]["relay_base"][
                "throughput_ratio"
            ]
            / summaries["original"]["native_reference"]["methods"]["relay_base"][
                "throughput_ratio"
            ]
        )
        cell["native_retention_interaction"] = dict(
            ratio=point,
            ci95=np.quantile(
                retentions[cell["style"]] / retentions["original"], [0.025, 0.975]
            ).tolist(),
        )
        scored.extend(dict(**r, study_comparison=cell["style"]) for r in rows)
    result = dict(
        status="complete",
        results=cells,
        input_sha256=inputs,
        scorer=name,
        scorer_provenance_sha256=digest(provenance_path),
        scope=protocol["scope"],
        analysis=protocol["analysis"],
    )
    (root / "instruction-style-summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (root / "instruction-style-scored.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in scored)
    )
    print(json.dumps(cells, indent=2))


if __name__ == "__main__":
    main()
