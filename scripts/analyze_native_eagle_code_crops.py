"""Audit paired untrained and fitted native code interfaces at four layer sets."""

import json
from pathlib import Path

import yaml
from score_native_code_study import execute

from relayspec.ar_paper_evidence import summarize
from relayspec.cached_fit_evidence import digest
from relayspec.code_eval import extract_code


def main():
    root = Path("reports/autoresearch-20260907")
    run = root / "run-28571"
    wave_path = Path("configs/autoresearch/20260907/wave76.json")
    specs = json.loads(wave_path.read_text())["lanes"]
    job = next(
        j
        for j in json.loads((root / "jobs.json").read_text())["jobs"]
        if j["id"] == 28571
    )
    assert (run / "source-commit.txt").read_text().strip() == job["source_commit"]
    inputs = {
        str(p): digest(p)
        for p in [
            wave_path,
            run / "source-commit.txt",
            Path(__file__),
            Path("scripts/score_native_code_study.py"),
            Path("src/relayspec/code_eval.py"),
        ]
    }
    results = []
    cached = {}
    common_ids = None
    for i, spec in enumerate(specs):
        lane = run / f"lane{i}"
        paths = [
            run / f"lane{i}-status.json",
            run / f"lane{i}-spec.json",
            lane / "transform.json",
            lane / "mapper-campaign.json",
            lane / "benchmark-rank0.jsonl",
            lane / "config.yaml",
        ]
        status, executed, transform, campaign = [
            json.loads(p.read_text()) for p in paths[:4]
        ]
        assert status["status"] == "pass" and executed == spec
        assert (
            transform["parent_sha256"] == spec["checkpoint_sha256"][spec["checkpoint"]]
        )
        assert transform["column_selection"]["selected_taps"] == spec["selected_taps"]
        assert transform["column_selection"]["selected_blocks"] == [
            [1, 9, 17, 25, 33].index(t) for t in spec["selected_taps"]
        ]
        for cp, sha in spec["checkpoint_sha256"].items():
            found = [
                v for v in campaign["variants"].values() if v["checkpoint_path"] == cp
            ]
            assert len(found) == 1 and found[0]["checkpoint_sha256"] == sha
        config = yaml.safe_load(Path(spec["config"]).read_text())
        actual = yaml.safe_load(paths[-1].read_text())
        for key in ["generation", "target", "proposer", "native_target_proposer"]:
            assert actual[key] == config[key]
        manifest = Path(config["benchmark"]["manifest_path"])
        records = {
            r["problem_id"]: r for r in json.loads(manifest.read_text())["records"]
        }
        rows = [json.loads(s) for s in paths[4].read_text().splitlines()]
        ids = set(records)
        methods = {
            "native_target_eagle3",
            "relay_base",
            "relay_cropped",
            "relay_fitted",
        }
        assert len(ids) == 16 and len(rows) == 64
        assert {(r["problem_id"], r["method"], r["repetition"]) for r in rows} == {
            (p, m, 0) for p in ids for m in methods
        }
        if common_ids is None:
            common_ids = ids
        assert ids == common_ids
        for r in rows:
            assert r["reference_answer"] == records[r["problem_id"]]["answer"]
            code = extract_code(r["completion"])
            tests = r["reference_answer"]["tests"]
            key = json.dumps([code, tests], sort_keys=True)
            if key not in cached:
                cached[key] = execute(code, tests)
            r["correct"] = cached[key]["passed"]
        results.append(
            dict(
                layers=spec["selected_taps"],
                native_reference=summarize(rows, reference="native_target_eagle3"),
                crop_reference=summarize(rows, reference="relay_cropped"),
                cap_counts={
                    m: sum(r["output_tokens"] >= 1024 for r in rows if r["method"] == m)
                    for m in methods
                },
            )
        )
        scored = lane / "code-scored.jsonl"
        scored.write_text("".join(json.dumps(r) + "\n" for r in rows))
        inputs.update(
            {
                str(p): digest(p)
                for p in [*paths, manifest, Path(spec["config"]), scored]
            }
        )
    out = dict(
        results=results,
        input_sha256=inputs,
        scope="Same16 exposed MBPP questions across four layer-set comparisons. Published base assertions in isolated Bubblewrap harness,1024token cap. Paired question intervals do not adjust for selection or multiple comparisons. Not64 independent questions or a quality noninferiority test.",
    )
    (root / "native-eagle-code-crops-summary.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    for c in results:
        print(
            c["layers"],
            c["crop_reference"]["methods"]["relay_fitted"],
            c["native_reference"]["methods"]["relay_cropped"]["throughput_ratio"],
        )


if __name__ == "__main__":
    main()
