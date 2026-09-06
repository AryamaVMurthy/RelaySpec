"""Audit paired cross-domain controls for released-column initialization."""

import json
from pathlib import Path

from relayspec.ar_paper_evidence import summarize
from relayspec.cached_fit_evidence import digest


def analyze(job_id, wave_number, output):
    root = Path("reports/autoresearch-20260907")
    run = root / f"run-{job_id}"
    wave_path = Path(f"configs/autoresearch/20260907/wave{wave_number}.json")
    wave = json.loads(wave_path.read_text())
    job = next(
        j
        for j in json.loads((root / "jobs.json").read_text())["jobs"]
        if j["id"] == job_id
    )
    assert (run / "source-commit.txt").read_text().strip() == job["source_commit"]
    inputs = {
        str(wave_path): digest(wave_path),
        str(run / "source-commit.txt"): digest(run / "source-commit.txt"),
    }
    results = []
    for i, (spec, task) in enumerate(
        zip(wave["lanes"], ["GSM8K", "MATH", "Code", "Dialogue"])
    ):
        lane = run / f"lane{i}"
        paths = [
            run / f"lane{i}-status.json",
            run / f"lane{i}-spec.json",
            lane / "transform.json",
            lane / "mapper-campaign.json",
            lane / "benchmark-rank0.jsonl",
        ]
        status, executed, transform, campaign = [
            json.loads(p.read_text()) for p in paths[:4]
        ]
        assert status["status"] == "pass" and executed == spec
        assert (
            transform["parent_sha256"] == spec["checkpoint_sha256"][spec["checkpoint"]]
        )
        assert transform["column_selection"]["selected_taps"] == [25, 33]
        assert transform["column_selection"]["selected_blocks"] == [3, 4]
        for cp, sha in spec["checkpoint_sha256"].items():
            found = [v for v in campaign["variants"].values() if v["checkpoint"] == cp]
            assert len(found) == 1 and found[0]["sha256"] == sha
        rows = [json.loads(s) for s in paths[-1].read_text().splitlines()]
        methods = {"relay_base", "relay_cropped", *spec["controls"]}
        ids = {r["problem_id"] for r in rows}
        assert len(ids) == (16 if task == "Dialogue" else 8)
        assert len(rows) == len(ids) * len(methods)
        assert {(r["problem_id"], r["method"]) for r in rows} == {
            (p, m) for p in ids for m in methods
        }
        summary = summarize(rows, reference="relay_base")
        crop = summarize(rows, reference="relay_cropped")
        results.append(
            dict(
                task=task,
                full_native_reference=summary,
                cropped_reference=crop,
                cap_counts={
                    m: sum(r["output_tokens"] == 512 for r in rows if r["method"] == m)
                    for m in methods
                },
            )
        )
        inputs.update({str(p): digest(p) for p in paths})
        print(
            task,
            {m: round(v["throughput_ratio"], 4) for m, v in summary["methods"].items()},
        )
    scope = "Eight exposed development requests per workload, two turns per dialogue conversation. Seed1729. Full reference is repacked released native projection. Same-worker comparisons retain untrained cropping and all four initialized/random16/128-record fits. Paired request intervals cluster dialogue turns. No fresh confirmation, code execution or dialogue quality claim."
    Path(output).write_text(
        json.dumps(dict(input_sha256=inputs, results=results, scope=scope), indent=2)
        + "\n"
    )
    return dict(input_sha256=inputs, results=results, scope=scope)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=int, default=28533)
    parser.add_argument("--wave", type=int, default=43)
    parser.add_argument(
        "--output",
        default="reports/autoresearch-20260907/native-initialization-transfer-summary.json",
    )
    args = parser.parse_args()
    analyze(args.job, args.wave, args.output)
