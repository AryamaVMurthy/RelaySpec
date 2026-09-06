"""Audit frozen native-code shards and execute base tests in Bubblewrap."""

import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path

import yaml

from relayspec.ar_paper_evidence import summarize
from relayspec.cached_fit_evidence import digest
from relayspec.code_eval import _limit_child, extract_code
from relayspec.paired_accuracy import paired_accuracy_interval


def execute(code, tests, timeout=5):
    with tempfile.TemporaryDirectory(prefix="relayspec-code-sandbox-") as directory:
        root = Path(directory)
        candidate = root / "candidate.py"
        candidate.write_text(code + "\n\n" + "\n".join(tests) + "\n")
        with (
            (root / "stdout").open("w") as stdout,
            (root / "stderr").open("w") as stderr,
        ):
            process = subprocess.Popen(
                [
                    "bwrap",
                    "--unshare-all",
                    "--die-with-parent",
                    "--clearenv",
                    "--ro-bind",
                    "/usr",
                    "/usr",
                    "--ro-bind",
                    "/lib",
                    "/lib",
                    "--ro-bind",
                    "/lib64",
                    "/lib64",
                    "--proc",
                    "/proc",
                    "--dev",
                    "/dev",
                    "--tmpfs",
                    "/tmp",
                    "--ro-bind",
                    str(candidate),
                    "/tmp/candidate.py",
                    "--chdir",
                    "/tmp",
                    "--setenv",
                    "PATH",
                    "/usr/bin",
                    "/usr/bin/python3",
                    "-I",
                    "/tmp/candidate.py",
                ],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                preexec_fn=lambda: _limit_child(timeout),
            )
            try:
                process.wait(timeout=timeout + 2)
                status = "pass" if process.returncode == 0 else "fail"
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                status = "timeout"
        return dict(
            passed=status == "pass",
            status=status,
            returncode=process.returncode,
            stdout=(root / "stdout").read_text()[-2000:],
            stderr=(root / "stderr").read_text()[-2000:],
        )


def main():
    root = Path("reports/autoresearch-20260907/run-28529")
    protocol_path = Path("configs/autoresearch/20260907/native-code-v1/protocol.json")
    wave_path = Path("configs/autoresearch/20260907/wave39.json")
    protocol = json.loads(protocol_path.read_text())
    specs = json.loads(wave_path.read_text())["lanes"]
    inputs = {
        str(p): digest(p)
        for p in [
            protocol_path,
            wave_path,
            Path(__file__),
            Path("src/relayspec/code_eval.py"),
        ]
    }
    ledger = json.loads(Path("reports/autoresearch-20260907/jobs.json").read_text())
    job = next(j for j in ledger["jobs"] if j["id"] == 28529)
    source_path = root / "source-commit.txt"
    if source_path.read_text().strip() != job["source_commit"]:
        raise ValueError("Source snapshot differs from recorded job")
    inputs[str(source_path)] = digest(source_path)
    all_rows = []
    expected_records = {}
    for i, spec in enumerate(specs):
        paths = [
            root / f"lane{i}-status.json",
            root / f"lane{i}-spec.json",
            root / f"lane{i}/mapper-campaign.json",
            root / f"lane{i}/benchmark-rank0.jsonl",
        ]
        status, executed, campaign = [json.loads(p.read_text()) for p in paths[:3]]
        if (
            status["status"] != "pass"
            or executed != spec
            or spec["checkpoint_sha256"] != protocol["checkpoints"]
        ):
            raise ValueError("Missing or undeclared frozen lane")
        cfg = yaml.safe_load(Path(spec["config"]).read_text())
        manifest_path = Path(cfg["benchmark"]["manifest_path"])
        records = json.loads(manifest_path.read_text())["records"]
        expected_records.update({r["problem_id"]: r for r in records})
        for cp, sha in protocol["checkpoints"].items():
            found = [v for v in campaign["variants"].values() if v["checkpoint"] == cp]
            if len(found) != 1 or found[0]["sha256"] != sha:
                raise ValueError("Mapper fingerprint mismatch")
        rows = [json.loads(line) for line in paths[3].read_text().splitlines()]
        methods = {"native_target_dflash", "relay_base", *spec["candidates"]}
        expected = {(r["problem_id"], m) for r in records for m in methods}
        if (
            len(rows) != 40
            or {(r["problem_id"], r["method"]) for r in rows} != expected
        ):
            raise ValueError("Missing or duplicate request/method output")
        for r in rows:
            if (
                r["reference_answer"] != expected_records[r["problem_id"]]["answer"]
                or r["benchmark"] != "mbpp"
                or r["repetition"] != 0
            ):
                raise ValueError("Mismatched tests or benchmark")
        all_rows.extend(rows)
        inputs.update(
            {str(p): digest(p) for p in [*paths, Path(spec["config"]), manifest_path]}
        )
    if (
        set(expected_records) != set(protocol["question_ids"])
        or len(expected_records) != 32
    ):
        raise ValueError("Frozen sample mismatch")
    cached = {}
    scored = []
    for row in all_rows:
        code = extract_code(row["completion"])
        tests = row["reference_answer"]["tests"]
        key = json.dumps([code, tests], sort_keys=True)
        if key not in cached:
            cached[key] = execute(code, tests)
        scored.append(
            dict(
                problem_id=row["problem_id"],
                method=row["method"],
                code=code,
                tests=tests,
                output_hash=row["output_hash"],
                output_tokens=row["output_tokens"],
                **cached[key],
            )
        )
    (root / "code-scored.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in scored)
    )
    reference = {
        r["problem_id"]: r for r in scored if r["method"] == "native_target_dflash"
    }
    results = {}
    for method in sorted({r["method"] for r in scored}):
        rows = sorted(
            [r for r in scored if r["method"] == method], key=lambda r: r["problem_id"]
        )
        results[method] = dict(
            passed=sum(r["passed"] for r in rows),
            requests=32,
            code_matches=sum(
                r["code"] == reference[r["problem_id"]]["code"] for r in rows
            ),
            token_matches=sum(
                r["output_hash"] == reference[r["problem_id"]]["output_hash"]
                for r in rows
            ),
            capped=sum(r["output_tokens"] >= 1024 for r in rows),
            paired_accuracy=paired_accuracy_interval(
                [r["passed"] for r in rows],
                [reference[r["problem_id"]]["passed"] for r in rows],
            ),
        )
    inputs[str(root / "code-scored.jsonl")] = digest(root / "code-scored.jsonl")
    summary = dict(
        python_version=subprocess.check_output(
            ["/usr/bin/python3", "--version"], text=True
        ).strip(),
        bubblewrap_version=subprocess.check_output(
            ["bwrap", "--version"], text=True
        ).strip(),
        input_sha256=inputs,
        results=results,
        throughput=summarize(all_rows, reference="native_target_dflash"),
        unique_code_test_pairs=len(cached),
        scorer="Published base assertions, first Python fence extraction, isolated system Python3 via Bubblewrap,5second test timeout. Not EvalPlus extended tests.",
        scope=protocol["exposure"]
        + " All32 tasks and five frozen methods retained. Paired accuracy uncertainty does not establish1pp noninferiority.",
    )
    Path("reports/autoresearch-20260907/native-code-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
