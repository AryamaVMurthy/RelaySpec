"""Collect terminal resumed jobs, audit them, and promote the composition pilot."""

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/aryamavmurthy/work/RelaySpec-scaling")
SCORING = Path("/home/aryamavmurthy/work/RelaySpec")
PYTHON = str(SCORING / ".venv/bin/python")
BASE = ROOT / "reports/mapper-scaling-20260905"
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def command(args, *, cwd=ROOT, env=ENV):
    result = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"{shlex.join(map(str, args))}\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}"
        )
    return result.stdout


def read(path):
    return json.loads(path.read_text())


def collect(job):
    run = ROOT / job["local"]
    run.mkdir(parents=True, exist_ok=True)
    command(
        [
            "rsync",
            "-az",
            "--exclude=source-snapshot.tar.gz",
            "turing:" + job["remote"] + "/",
            str(run) + "/",
        ]
    )
    command(
        [PYTHON, "scripts/analyze_controlled_run.py", str(run)],
        cwd=SCORING,
        env={**os.environ, "PYTHONPATH": "src:vendor/qwen-score-deps"},
    )
    return run


def main():
    done = set()
    confirmation = BASE / "resumed-confirmation"
    composition = BASE / "resumed-composition-evaluation"
    while True:
        ledger = read(confirmation / "jobs-full.json")
        jobs = [("confirmation", j) for j in ledger["jobs"]]
        comp_ledger = read(composition / "jobs-pilot.json")
        jobs += [("composition", j) for j in comp_ledger["jobs"]]
        if (composition / "jobs-full.json").exists():
            jobs += [
                ("composition", j) for j in read(composition / "jobs-full.json")["jobs"]
            ]
        ids = ",".join(str(j["id"]) for _, j in jobs)
        raw = command(
            [
                "ssh",
                "turing",
                f"sacct -X -j {ids} --noheader --parsable2 --format=JobID,State,Elapsed",
            ]
        )
        states = {}
        for line in raw.splitlines():
            fields = line.split("|")
            if len(fields) >= 3:
                states[int(fields[0])] = (fields[1], fields[2])
        print(
            json.dumps({"time": time.time(), "jobs": states, "audited": sorted(done)}),
            flush=True,
        )
        for kind, job in jobs:
            jid = job["id"]
            if jid in done:
                continue
            state = states.get(jid, ("UNKNOWN", ""))[0]
            if state in ["PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "UNKNOWN"]:
                continue
            if state != "COMPLETED":
                raise RuntimeError(
                    f"Job {jid} ended in {state}; inspect before retrying"
                )
            run = collect(job)
            if kind == "confirmation":
                family = job["family"]
                command(
                    [
                        PYTHON,
                        "scripts/audit_eagle_quality.py",
                        "--protocol",
                        f"configs/submission/confirmation-gsm8k-20260906/frozen-dense-v1/{family}-protocol.json",
                        "--stage",
                        "full",
                        "--runs",
                        str(run),
                        "--ledger",
                        str(confirmation / "jobs-full.json"),
                        "--scoring-repo",
                        str(SCORING),
                        "--pilot-result",
                        str(confirmation / f"{family}-pilot-audit.json"),
                        "--output",
                        str(confirmation / f"{family}-full-audit.json"),
                    ]
                )
            else:
                args = [
                    PYTHON,
                    "scripts/audit_composition_evaluation.py",
                    "--run",
                    str(run),
                    "--stage",
                    job["stage"],
                    "--ledger",
                    str(composition / f"jobs-{job['stage']}.json"),
                    "--output",
                    str(composition / f"{job['stage']}-audit.json"),
                ]
                if job["stage"] == "full":
                    args += ["--pilot-audit", str(composition / "pilot-audit.json")]
                command(args)
                if (
                    job["stage"] == "pilot"
                    and not (composition / "jobs-full.json").exists()
                ):
                    project = "/home/aryama.murthy/relayspec-composition-eval-v1"
                    config = "configs/submission/scaling/composition-small-v1/evaluation-full.yaml"
                    name = "rs-comp-decode-full"
                    exports = {
                        "RELAYSPEC_PROJECT_DIR": project,
                        "CAMPAIGN_CONFIG": config,
                        "CAMPAIGN_CHECK_DUPLICATE": "0",
                        "CAMPAIGN_TIMEOUT": "4440",
                    }
                    args = [
                        "sbatch",
                        "--parsable",
                        "--time=01:15:00",
                        "--job-name=" + name,
                        "--export=ALL,"
                        + ",".join(k + "=" + v for k, v in exports.items()),
                        "slurm/mapper_campaign.sbatch",
                    ]
                    submitted = command(
                        [
                            "ssh",
                            "turing",
                            "cd " + shlex.quote(project) + " && " + shlex.join(args),
                        ]
                    )
                    new_id = int(submitted.strip().split(";")[0])
                    full = {
                        "source_commit": comp_ledger["source_commit"],
                        "max_gpus": 4,
                        "jobs": [
                            {
                                "id": new_id,
                                "stage": "full",
                                "config": config,
                                "remote": f"{project}/outputs/{name}-{new_id}",
                                "local": str(
                                    (composition / f"run-{new_id}").relative_to(ROOT)
                                ),
                            }
                        ],
                    }
                    (composition / "jobs-full.json").write_text(
                        json.dumps(full, indent=2) + "\n"
                    )
                    print(f"SUBMITTED composition full {new_id}", flush=True)
            done.add(jid)
            print(f"AUDITED {kind} {jid}", flush=True)
        status = {
            "updated_unix": time.time(),
            "audited_jobs": sorted(done),
            "states": states,
            "scope": "Live scheduler polling; full promotion only after raw scoring and declared pilot audit.",
        }
        (BASE / "resumed-watch-status.json").write_text(
            json.dumps(status, indent=2) + "\n"
        )
        if (composition / "jobs-full.json").exists():
            final = read(composition / "jobs-full.json")["jobs"][0]["id"]
            if final in done and all(j["id"] in done for j in ledger["jobs"]):
                print("ALL_RESUMED_EVALUATIONS_AUDITED", flush=True)
                return
        time.sleep(40)


if __name__ == "__main__":
    main()
