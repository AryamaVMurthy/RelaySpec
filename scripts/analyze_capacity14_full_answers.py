"""Audit and score the frozen 14B full-answer capacity comparison."""

import argparse
import json
import os
from pathlib import Path

import yaml
from aggregate_results import build_math_scorer

from relayspec.ar_paper_evidence import summarize
from relayspec.cached_fit_evidence import digest
from relayspec.paired_accuracy import paired_accuracy_interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring-repo", type=Path, required=True)
    args = parser.parse_args()
    repo = Path.cwd()
    root = repo / "reports/autoresearch-20260907"
    base = repo / "configs/autoresearch/20260907"
    protocol_path = base / "capacity14-full-answer-v1/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    manifest = base / "sampling-breadth-v1/manifest32.json"
    assert digest(manifest) == protocol["manifest_sha256"]
    expected_ids = {
        r["problem_id"] for r in json.loads(manifest.read_text())["records"]
    }
    ledger = json.loads((root / "jobs.json").read_text())["jobs"]
    inputs = {str(p.relative_to(repo)): digest(p) for p in [protocol_path, manifest]}
    cells = {}
    for number in protocol["waves"]:
        jobs = [j for j in ledger if j["wave"] == number]
        assert len(jobs) == 1
        job = jobs[0]
        run = root / f"run-{job['id']}"
        source = run / "source-commit.txt"
        assert source.read_text().strip() == job["source_commit"]
        wave = base / f"wave{number}.json"
        inputs.update({str(p.relative_to(repo)): digest(p) for p in [source, wave]})
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
            for field in ["generation", "target", "proposer", "source_trunk"]:
                assert config[field] == actual[field]
            shard = repo / config["benchmark"]["manifest_path"]
            records = {
                r["problem_id"]: r for r in json.loads(shard.read_text())["records"]
            }
            methods = {"native_ar", "relay_base", *spec["candidates"]}
            assert len(methods) == 5 and set(actual["benchmark"]["methods"]) == methods
            campaign_path = lane / "mapper-campaign.json"
            campaign = json.loads(campaign_path.read_text())
            for cp, sha in spec["checkpoint_sha256"].items():
                assert protocol["checkpoints"][cp] == sha
                found = [
                    v
                    for v in campaign["variants"].values()
                    if v.get("checkpoint", v.get("checkpoint_path")) == cp
                ]
                assert (
                    len(found) == 1
                    and found[0].get("sha256", found[0].get("checkpoint_sha256")) == sha
                )
            path = lane / "benchmark-rank0.jsonl"
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            assert len(rows) == 40 and {
                (r["problem_id"], r["method"], r["repetition"]) for r in rows
            } == {(p, m, 0) for p in records for m in methods}
            for r in rows:
                assert (
                    r["benchmark"] == "gsm8k"
                    and r["reference_answer"] == records[r["problem_id"]]["answer"]
                )
                r["study_family"] = spec["family"]
            cells.setdefault(spec["family"], []).extend(rows)
            inputs.update(
                {
                    str(p.relative_to(repo)): digest(p)
                    for p in [
                        status,
                        executed,
                        config_path,
                        lane / "config.yaml",
                        shard,
                        campaign_path,
                        path,
                        lane / "completion-gate.json",
                    ]
                }
            )
    assert sum(map(len, cells.values())) == 320 and set(cells) == set(
        protocol["families"]
    )
    os.chdir(args.scoring_repo)
    provenance_path = Path("reports/ar-revision-20260905/scorer-provenance.json")
    provenance = json.loads(provenance_path.read_text())
    assert all(digest(Path(p)) == sha for p, sha in provenance["sha256"].items())
    scorer, name = build_math_scorer(Path("vendor/qwen-math/evaluation"))
    cache = {}
    results = []
    for family, rows in sorted(cells.items()):
        assert {r["problem_id"] for r in rows} == expected_ids
        for r in rows:
            key = (r["completion"], str(r["reference_answer"]), r["benchmark"])
            if key not in cache:
                cache[key] = scorer(*key)
            r.update(cache[key])
        lookup = {(r["problem_id"], r["method"]): r for r in rows}
        ids = sorted(expected_ids)
        methods = sorted({r["method"] for r in rows})
        quality = {
            m: paired_accuracy_interval(
                [lookup[p, m]["correct"] for p in ids],
                [lookup[p, "native_ar"]["correct"] for p in ids],
            )
            for m in methods
        }
        results.append(
            dict(
                family=family,
                ar_reference=summarize(rows, reference="native_ar"),
                dense2048_reference=summarize(rows, reference="relay_base"),
                factorized_reference=summarize(rows, reference="relay_factorized4096_n2048"),
                paired_accuracy_vs_ar=quality,
                cap_counts={
                    m: sum(r["output_tokens"] == 2048 for r in rows if r["method"] == m)
                    for m in methods
                },
            )
        )
    output = dict(
        status="complete",
        results=results,
        input_sha256=inputs,
        scorer=name,
        scorer_provenance_sha256=digest(provenance_path),
        scope=protocol["scope"],
        analysis=protocol["analysis"],
    )
    (root / "capacity14-full-answer-summary.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    (root / "capacity14-full-answer-scored.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for rows in cells.values() for r in rows)
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
