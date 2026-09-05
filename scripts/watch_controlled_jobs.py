"""Collect and analyze this fixed job list for at most 24 hours. No submissions."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--jobs",
        type=Path,
        default=Path("reports/controlled-scaling-20260905/jobs.json"),
    )
    args = parser.parse_args()
    base = args.jobs.parent
    jobs = json.loads(args.jobs.read_text())["jobs"]
    status_path = base / "collector-status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    deadline = time.monotonic() + 86400
    terminal = {
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
    }
    while time.monotonic() < deadline:
        for job in jobs:
            key = str(job["id"])
            if status.get(key, {}).get("collected") or status.get(key, {}).get(
                "terminal_failure"
            ):
                continue
            try:
                result = subprocess.run(
                    [
                        "ssh",
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        "ConnectTimeout=10",
                        "turing",
                        f"sacct -j {int(key)} -X -n -P --format=State,Elapsed",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                line = result.stdout.strip().splitlines()[0]
                state, elapsed = line.split("|")[:2]
                status[key] = {
                    "state": state,
                    "elapsed": elapsed,
                    "checked_unix": time.time(),
                }
                if state.split()[0] in terminal:
                    status[key]["terminal_failure"] = True
                if state == "COMPLETED":
                    destination = Path(job["local"])
                    destination.mkdir(parents=True, exist_ok=True)
                    subprocess.run(
                        [
                            "rsync",
                            "-az",
                            "--exclude=*.pt",
                            "-e",
                            "ssh -o BatchMode=yes -o ConnectTimeout=10",
                            f"turing:{job['remote']}/",
                            str(destination) + "/",
                        ],
                        check=True,
                        timeout=180,
                    )
                    evaluations = (
                        sorted(destination.glob("evaluation-step-*"))
                        if job["kind"] == "scaling"
                        else [destination]
                    )
                    if not evaluations:
                        raise ValueError("completed job has no evaluations")
                    for evaluation in evaluations:
                        subprocess.run(
                            [
                                sys.executable,
                                "scripts/analyze_controlled_run.py",
                                str(evaluation),
                            ],
                            check=True,
                            timeout=1200,
                        )
                    status[key]["collected"] = True
            except (
                subprocess.SubprocessError,
                OSError,
                ValueError,
                IndexError,
            ) as error:
                status.setdefault(key, {})["last_error"] = str(error)
                print(f"job {key}: {error}", flush=True)
            status_path.write_text(json.dumps(status, indent=2) + "\n")
        if args.once or all(
            status.get(str(j["id"]), {}).get("collected")
            or status.get(str(j["id"]), {}).get("terminal_failure")
            for j in jobs
        ):
            return
        time.sleep(60)
    print("Collector stopped at its 24-hour deadline", flush=True)


if __name__ == "__main__":
    main()
