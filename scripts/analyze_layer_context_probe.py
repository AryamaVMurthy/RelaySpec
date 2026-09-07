"""Audit every declared layer/context fitting arm and report paired decoding."""

import argparse
import json
import os
from pathlib import Path

import yaml
from aggregate_results import build_math_scorer

from relayspec.ar_paper_evidence import summarize
from relayspec.cached_fit_evidence import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wave", type=int, required=True)
    p.add_argument("--scoring-repo", type=Path, required=True)
    a = p.parse_args()
    repo = Path.cwd()
    root = repo / "reports/autoresearch-20260907"
    ledger = json.loads((root / "jobs.json").read_text())["jobs"]
    job = next(j for j in ledger if j["wave"] == a.wave)
    run = root / f"run-{job['id']}"
    wave = repo / f"configs/autoresearch/20260907/wave{a.wave}.json"
    specs = json.loads(wave.read_text())["lanes"]
    inputs = {str(wave): digest(wave)}
    cells = []
    assert (run / "source-commit.txt").read_text().strip() == job["source_commit"]
    for i, s in enumerate(specs):
        lane = run / f"lane{i}"
        assert (
            json.loads((run / f"lane{i}-status.json").read_text())["status"] == "pass"
        )
        assert json.loads((run / f"lane{i}-spec.json").read_text()) == s
        config = yaml.safe_load((repo / s["config"]).read_text())
        actual = yaml.safe_load((lane / "config.yaml").read_text())
        manifest = repo / config["benchmark"]["manifest_path"]
        golds = {
            r["problem_id"]: r
            for r in json.loads(manifest.read_text())["records"][: s["requests"]]
        }
        assert actual["generation"]["max_new_tokens"] == s["max_new_tokens"]
        for k in ["target", "proposer"]:
            assert actual[k] == config[k]
        rows = [
            json.loads(x)
            for x in (lane / "benchmark-rank0.jsonl").read_text().splitlines()
        ]
        methods = set(actual["benchmark"]["methods"])
        assert len(rows) == len(golds) * len(methods)
        assert {(r["problem_id"], r["method"]) for r in rows} == {
            (pid, m) for pid in golds for m in methods
        }
        for row in rows:
            assert (
                row["reference_answer"] == golds[row["problem_id"]]["answer"]
                and row["repetition"] == 0
            )
        campaign = json.loads((lane / "mapper-campaign.json").read_text())["variants"]
        fit = None
        if s["transform"] == "fit_layer_context":
            fit = json.loads((lane / "layer-context-fit.json").read_text())
            assert fit["status"] == "pass" and fit["settings"] == s["layer_context_fit"]
            v = campaign["relay_layer_context"]
            assert v.get("checkpoint_sha256", v.get("sha256")) == fit["checkpoint_sha256"]
            selection = json.loads((lane / "layer-context-selection.json").read_text())
            assert (
                len(selection)
                == fit["settings"]["records"] + fit["settings"]["validation_records"]
            )
            assert all(
                len(r["positions"]) == 32 and len(set(r["positions"])) == 32
                for r in selection
            )
        else:
            for cp, sha in s["checkpoint_sha256"].items():
                assert any(
                    v.get("checkpoint_path", v.get("checkpoint")) == cp and v.get("checkpoint_sha256", v.get("sha256")) == sha
                    for v in campaign.values()
                )
        for path in [
            wave,
            manifest,
            repo / s["config"],
            run / "source-commit.txt",
            run / f"lane{i}-status.json",
            run / f"lane{i}-spec.json",
            *lane.glob("*.json"),
            lane / "benchmark-rank0.jsonl",
        ]:
            inputs[str(path.relative_to(repo))] = digest(path)
        cells.append(dict(lane=i, fit=fit, rows=rows))
    if specs[0]["transform"] == "fit_layer_context":
        selections = [
            digest(run / f"lane{i}/layer-context-selection.json") for i in range(4)
        ]
        assert len(set(selections)) == 1, "Arm data or sampled positions differ"
        initial = [c["fit"]["history"][0]["validation"] for c in cells[:3]]
        assert initial[0] == initial[1] == initial[2], (
            "Five-map arms did not share initialization"
        )
    os.chdir(a.scoring_repo)
    provenance = Path("reports/ar-revision-20260905/scorer-provenance.json")
    assert all(
        digest(Path(k)) == v
        for k, v in json.loads(provenance.read_text())["sha256"].items()
    )
    score, name = build_math_scorer(Path("vendor/qwen-math/evaluation"))
    memo = {}
    for cell in cells:
        for row in cell["rows"]:
            key = (row["completion"], str(row["reference_answer"]), row["benchmark"])
            if key not in memo:
                memo[key] = score(*key)
            row.update(memo[key])
        cell["throughput"] = summarize(cell["rows"], reference="relay_base")
        cell["quality"] = {
            m: dict(
                correct=sum(
                    bool(r["correct"]) for r in cell["rows"] if r["method"] == m
                ),
                capped=sum(
                    r["output_tokens"] >= specs[cell["lane"]]["max_new_tokens"]
                    for r in cell["rows"]
                    if r["method"] == m
                ),
            )
            for m in cell["throughput"]["methods"]
        }
    summary = dict(
        job=job["id"],
        wave=a.wave,
        cells=cells,
        input_sha256=inputs,
        scorer=name,
        scope="All declared arms retained. Development evaluation, no fresh confirmation or accuracy noninferiority claim. Training baseline budget differences retained.",
    )
    if specs[0]["transform"] == "existing":
        rows = [r for c in cells for r in c["rows"]]
        summary["pooled"] = summarize(rows, reference="relay_base")
        summary["matched_dense"] = summarize(rows, reference="relay_dense_context")
        summary["matched_context"] = summarize(rows, reference="relay_context_only")
    (root / f"layer-context-wave{a.wave}-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    for cell in cells:
        print(cell["lane"], cell["throughput"]["methods"], cell["quality"])


if __name__ == "__main__":
    main()
