"""Audit and score the four declared calibration-domain/capacity cells."""

import json
from pathlib import Path

import yaml
from analyze_compact_interface_data import analyze
from score_native_code_study import execute

from relayspec.ar_paper_evidence import summarize
from relayspec.cached_fit_evidence import digest
from relayspec.code_eval import extract_code

ROOT = Path("reports/autoresearch-20260907")


def main():
    output = ROOT / "code-calibration-pilot-summary.json"
    result = analyze(
        28604,
        94,
        output,
        cache_sha=[
            "9c18d186420ff569186fc1232b411cbb984ae05bbe09564a36f6179ceaccad19",
            "9c18d186420ff569186fc1232b411cbb984ae05bbe09564a36f6179ceaccad19",
            "a64f712baaa3ef459cc61cf0e9760f7a0bd83109cd89bc69b3070b28e3eff88c",
            "a64f712baaa3ef459cc61cf0e9760f7a0bd83109cd89bc69b3070b28e3eff88c",
        ],
    )
    specs = json.loads(Path("configs/autoresearch/20260907/wave94.json").read_text())[
        "lanes"
    ]
    memo, scored, cross_rows = {}, [], []
    for i, (cell, spec) in enumerate(zip(result["results"], specs, strict=True)):
        cfg_path = Path(spec["config"])
        cfg = yaml.safe_load(cfg_path.read_text())
        manifest = Path(cfg["benchmark"]["manifest_path"])
        expected = {
            r["problem_id"]: r for r in json.loads(manifest.read_text())["records"]
        }
        rows = [
            json.loads(s)
            for s in (ROOT / f"run-28604/lane{i}/benchmark-rank0.jsonl")
            .read_text()
            .splitlines()
        ]
        assert {r["problem_id"] for r in rows} == set(expected)
        native = {
            r["problem_id"]: r for r in rows if r["method"] == "native_target_eagle3"
        }
        domain = "math" if i < 2 else "code"
        layers = 2 + i % 2
        cell.update(
            domain=domain,
            layers=layers,
            native_reference=summarize(rows, reference="native_target_eagle3"),
        )
        quality = {}
        for row in rows:
            assert row["reference_answer"] == expected[row["problem_id"]]["answer"]
            assert row["benchmark"] == "mbpp" and row["repetition"] == 0
            code = extract_code(row["completion"])
            tests = row["reference_answer"]["tests"]
            key = json.dumps([code, tests], sort_keys=True)
            if key not in memo:
                memo[key] = execute(code, tests)
            sc = dict(
                lane=i,
                domain=domain,
                layers=layers,
                method=row["method"],
                problem_id=row["problem_id"],
                output_hash=row["output_hash"],
                output_tokens=row["output_tokens"],
                **memo[key],
            )
            scored.append(sc)
            q = quality.setdefault(
                row["method"], dict(passed=0, requests=0, capped=0, token_matches=0)
            )
            q["passed"] += bool(sc["passed"])
            q["requests"] += 1
            q["capped"] += row["output_tokens"] >= 1024
            q["token_matches"] += (
                row["output_hash"] == native[row["problem_id"]]["output_hash"]
            )
            if row["method"] == "relay_reduced":
                cross_rows.append(dict(row, method=f"{domain}_{layers}layers"))
        cell["quality"] = quality
        result["input_sha256"].update({str(p): digest(p) for p in [cfg_path, manifest]})
    result["cross_worker_comparison"] = summarize(cross_rows, reference="math_2layers")
    result["scope"] = (
        "Four predeclared math/code by two/three-layer fits,512 records128 updates. Eight shared exposed MBPP tasks,1024token cap. Record counts and updates matched but token exposure and data quality differ. Cross-worker timings may include device variation; frozen within-worker comparison required before strong interpretation. Base assertions in isolated Python, not EvalPlus. No fresh confirmation or quality noninferiority claim."
    )
    score_path = ROOT / "code-calibration-pilot-scored.jsonl"
    score_path.write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in scored))
    result["input_sha256"].update(
        {
            str(p): digest(p)
            for p in [
                Path(__file__),
                Path("scripts/analyze_compact_interface_data.py"),
                Path("scripts/score_native_code_study.py"),
                Path("src/relayspec/code_eval.py"),
                score_path,
            ]
        }
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    for cell in result["results"]:
        print(
            cell["domain"],
            cell["layers"],
            cell["native_reference"]["methods"]["relay_reduced"],
            cell["quality"],
        )


if __name__ == "__main__":
    main()
