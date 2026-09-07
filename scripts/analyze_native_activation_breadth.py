"""Audit native activation/weight compression and score every frozen output."""

import argparse
import json
import os
from pathlib import Path

import yaml
from aggregate_results import build_math_scorer
from score_native_code_study import execute

from relayspec.ar_paper_evidence import summarize
from relayspec.cached_fit_evidence import digest
from relayspec.code_eval import extract_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring-repo", type=Path, required=True)
    args = parser.parse_args()
    repo = Path.cwd()
    root = repo / "reports/autoresearch-20260907"
    protocol = (
        repo
        / "configs/autoresearch/20260907/native-activation-breadth-v1/protocol.json"
    )
    declaration = json.loads(protocol.read_text())
    ledger = json.loads((root / "jobs.json").read_text())["jobs"]
    inputs = {str(protocol.relative_to(repo)): digest(protocol)}
    cells = []
    for wave_number in declaration["waves"]:
        entry = next(j for j in ledger if j["wave"] == wave_number)
        run = root / f"run-{entry['id']}"
        wave = repo / f"configs/autoresearch/20260907/wave{wave_number}.json"
        source = run / "source-commit.txt"
        assert source.read_text().strip() == entry["source_commit"]
        specs = json.loads(wave.read_text())["lanes"]
        assert len(specs) == 4
        for p in [wave, source]:
            inputs[str(p.relative_to(repo))] = digest(p)
        for i, spec in enumerate(specs):
            lane = run / f"lane{i}"
            assert (
                json.loads((run / f"lane{i}-status.json").read_text())["status"]
                == "pass"
            )
            assert json.loads((run / f"lane{i}-spec.json").read_text()) == spec
            cfg = yaml.safe_load((repo / spec["config"]).read_text())
            actual = yaml.safe_load((lane / "config.yaml").read_text())
            for key in ["generation", "target", "proposer", "native_target_proposer"]:
                assert cfg[key] == actual[key]
            manifest = repo / cfg["benchmark"]["manifest_path"]
            golds = {
                r["problem_id"]: r for r in json.loads(manifest.read_text())["records"]
            }
            assert len(golds) == spec["requests"] == 8
            native = "native_target_" + spec["family"]
            methods = {
                native,
                "relay_base",
                "relay_two",
                "relay_weight1536",
                "relay_activation1536",
            }
            assert set(actual["benchmark"]["methods"]) == methods
            transform = json.loads((lane / "transform.json").read_text())
            assert (
                transform["parent_sha256"]
                == spec["checkpoint_sha256"][spec["checkpoint"]]
            )
            campaign = json.loads((lane / "mapper-campaign.json").read_text())[
                "variants"
            ]
            for method, cp in {
                "relay_base": spec["checkpoint"],
                **spec["candidates"],
            }.items():
                v = campaign[method]
                assert v.get("checkpoint", v.get("checkpoint_path")) == cp
                assert (
                    v.get("sha256", v.get("checkpoint_sha256"))
                    == spec["checkpoint_sha256"][cp]
                )
            assert (
                campaign["relay_activation1536"]["parameters"]
                == campaign["relay_weight1536"]["parameters"]
                == 37748736
            )
            assert campaign["relay_two"]["parameters"] == 33554432
            path = lane / "benchmark-rank0.jsonl"
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            assert len(rows) == 40 and {
                (r["problem_id"], r["method"], r["repetition"]) for r in rows
            } == {(p, m, 0) for p in golds for m in methods}
            for r in rows:
                assert r["reference_answer"] == golds[r["problem_id"]]["answer"]
                assert r["benchmark"] == golds[r["problem_id"]]["benchmark"]
            for p in [
                path,
                manifest,
                repo / spec["config"],
                lane / "config.yaml",
                lane / "transform.json",
                lane / "mapper-campaign.json",
                run / f"lane{i}-status.json",
                run / f"lane{i}-spec.json",
                lane / "completion-gate.json",
            ]:
                inputs[str(p.relative_to(repo))] = digest(p)
            cells.append(
                dict(
                    family=spec["family"],
                    workload=rows[0]["benchmark"],
                    native=native,
                    rows=rows,
                    max_new_tokens=spec["max_new_tokens"],
                )
            )
    grouped = {}
    for cell in cells:
        key = (cell["family"], cell["workload"])
        if key not in grouped:
            grouped[key] = {**cell, "rows": []}
        grouped[key]["rows"].extend(cell["rows"])
    cells = list(grouped.values())
    assert len(cells) == 4 and sum(len(c["rows"]) for c in cells) == 640
    for cell in cells:
        rows = cell["rows"]
        assert len({r["problem_id"] for r in rows}) == 32
        assert len({(r["problem_id"], r["method"]) for r in rows}) == 160
    for workload in ("gsm8k", "mbpp"):
        pairs = [c for c in cells if c["workload"] == workload]
        assert {r["problem_id"] for r in pairs[0]["rows"]} == {
            r["problem_id"] for r in pairs[1]["rows"]
        }
    os.chdir(args.scoring_repo)
    provenance = Path("reports/ar-revision-20260905/scorer-provenance.json")
    assert all(
        digest(Path(p)) == sha
        for p, sha in json.loads(provenance.read_text())["sha256"].items()
    )
    scorer, scorer_name = build_math_scorer(Path("vendor/qwen-math/evaluation"))
    cached = {}
    scored = []
    for cell in cells:
        rows = cell.pop("rows")
        for r in rows:
            if r["benchmark"] == "mbpp":
                code = extract_code(r["completion"])
                tests = r["reference_answer"]["tests"]
                key = json.dumps([code, tests])
                if key not in cached:
                    cached[key] = execute(code, tests)
                r["code_evaluation"] = cached[key]
                r["correct"] = cached[key]["passed"]
            else:
                key = (r["completion"], str(r["reference_answer"]), r["benchmark"])
                if key not in cached:
                    cached[key] = scorer(*key)
                r.update(cached[key])
        cell["native_reference"] = summarize(rows, reference=cell["native"])
        cell["weight_reference"] = summarize(rows, reference="relay_weight1536")
        cell["cap_counts"] = {
            m: sum(
                r["output_tokens"] == cell["max_new_tokens"]
                for r in rows
                if r["method"] == m
            )
            for m in cell["native_reference"]["methods"]
        }
        scored.extend(
            dict(**r, study_family=cell["family"], study_workload=cell["workload"])
            for r in rows
        )
    result = dict(
        results=cells,
        input_sha256=inputs,
        scorer=scorer_name,
        scorer_provenance_sha256=digest(provenance),
        scope=json.loads(protocol.read_text()),
    )
    (root / "native-activation-breadth-summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (root / "native-activation-breadth-scored.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in scored)
    )
    print(json.dumps(cells, indent=2))


if __name__ == "__main__":
    main()
