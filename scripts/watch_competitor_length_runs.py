"""Collect and audit the declared batch jobs; never launch or silently retry jobs."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/competitors-length-20260906"
SCORING = ROOT.parent / "RelaySpec"
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def command(args, **kwargs):
    return subprocess.run(
        args, check=True, text=True, capture_output=True, **kwargs
    ).stdout


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {}
    for job in json.loads((BASE / "jobs.json").read_text())["jobs"]:
        if job["kind"].endswith("context-full"):
            p = BASE / f"run-{job['id']}" / "audit.json"
            if p.exists():
                a = json.loads(p.read_text())
                if a["status"] == "pass":
                    series.setdefault(job["kind"], []).append((job["length"], a))
    if not series:
        return
    for family, rows in series.items():
        rows.sort()
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
        for method in rows[0][1]["summary"]["methods"]:
            xs = [n for n, _ in rows]
            axes[0].plot(
                xs,
                [a["summary"]["methods"][method]["tokens_per_second"] for _, a in rows],
                marker="o",
                label=method,
            )
            axes[1].plot(
                xs,
                [
                    a["sequence_metrics"][method]["mean_prefill_seconds"]
                    for _, a in rows
                ],
                marker="o",
            )
            axes[2].plot(
                xs,
                [
                    a["sequence_metrics"][method]["max_incremental_peak_gib"]
                    for _, a in rows
                ],
                marker="o",
            )
        for ax, title in zip(
            axes,
            [
                "End-to-end tokens/s",
                "Mean time to first token (s)",
                "Incremental peak allocated GiB",
            ],
            strict=True,
        ):
            ax.set_xscale("log", base=2)
            ax.set_xticks([4096, 8192, 16384, 32768], ["4K", "8K", "16K", "32K"])
            ax.set_xlabel("Exact input tokens")
            ax.set_title(title)
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=6)
        fig.suptitle(
            f"{family}: completed cells only ({len(rows)}/4), fixed 256-token output cap"
        )
        fig.tight_layout()
        fig.savefig(BASE / f"{family}.png", dpi=180)
        plt.close(fig)


def main():
    done = set()
    deadline = time.monotonic() + 8 * 3600
    while time.monotonic() < deadline:
        jobs = json.loads((BASE / "jobs.json").read_text())["jobs"]
        ids = ",".join(str(j["id"]) for j in jobs)
        raw = command(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "turing",
                f"sacct -X -j {ids} --noheader --parsable2 --format=JobID,State,Elapsed",
            ]
        )
        states = {
            int(f[0]): {"state": f[1], "elapsed": f[2]}
            for line in raw.splitlines()
            if len(f := line.split("|")) >= 3 and f[0].isdigit()
        }
        for job in jobs:
            jid = job["id"]
            state = states.get(jid, {}).get("state", "UNKNOWN")
            if jid in done or state in {
                "UNKNOWN",
                "PENDING",
                "RUNNING",
                "CONFIGURING",
                "COMPLETING",
            }:
                continue
            out = BASE / f"run-{jid}"
            out.mkdir(exist_ok=True)
            try:
                command(
                    [
                        "rsync",
                        "-az",
                        "--exclude=source-snapshot.tar.gz",
                        "turing:" + job["remote"] + "/",
                        str(out) + "/",
                    ]
                )
                if state == "COMPLETED" and job["kind"] != "setup":
                    command(
                        [
                            sys.executable,
                            "scripts/audit_competitor_length_run.py",
                            str(out),
                        ],
                        cwd=ROOT,
                        env=ENV,
                    )
                    if job["kind"] == "pard2-full":
                        command(
                            [
                                sys.executable,
                                str(ROOT / "scripts/analyze_controlled_run.py"),
                                str(out),
                            ],
                            cwd=SCORING,
                            env={
                                **ENV,
                                "PYTHONPATH": str(ROOT / "src")
                                + ":"
                                + str(SCORING / "vendor/qwen-score-deps"),
                            },
                        )
                states[jid]["artifact_status"] = (
                    "audited" if state == "COMPLETED" else "failed job retained"
                )
            except Exception as exc:
                states[jid]["artifact_status"] = "audit_or_collection_failed"
                states[jid]["error"] = str(exc)
                (out / "collection-error.txt").write_text(
                    str(exc) + "\n" + getattr(exc, "stderr", "")
                )
            done.add(jid)
        previous = (
            json.loads((BASE / "status.json").read_text())
            if (BASE / "status.json").exists()
            else {"jobs": {}}
        )
        for jid, state in states.items():
            previous["jobs"][str(jid)] = {**previous["jobs"].get(str(jid), {}), **state}
        previous.update(updated_unix=time.time(), terminal_processed=sorted(done))
        (BASE / "status.json").write_text(json.dumps(previous, indent=2) + "\n")
        plot()
        if len(done) == len(jobs):
            return
        time.sleep(30)
    raise TimeoutError(
        "Collection deadline reached; consult Slurm states before resuming"
    )


if __name__ == "__main__":
    main()
