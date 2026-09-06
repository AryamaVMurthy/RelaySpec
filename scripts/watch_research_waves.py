"""Collect declared autoresearch waves; never submit or retry experiments."""

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/autoresearch-20260907"


def main():
    done = set()
    deadline = time.monotonic() + 24 * 3600
    while time.monotonic() < deadline:
        jobs = json.loads((BASE / "jobs.json").read_text())["jobs"]
        status = {}
        for job in jobs:
            key = str(job["id"])
            if key in done:
                continue
            result = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=10",
                    "turing",
                    f"sacct -X -j {int(key)} --noheader --parsable2 --format=State,Elapsed",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            state, elapsed = result.stdout.strip().splitlines()[0].split("|")[:2]
            status[key] = {
                "state": state,
                "elapsed": elapsed,
                "checked_unix": time.time(),
            }
            if state in {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING"}:
                continue
            dest = BASE / f"run-{key}"
            dest.mkdir(exist_ok=True)
            subprocess.run(
                [
                    "rsync",
                    "-az",
                    "--exclude=*.pt",
                    "--exclude=source-snapshot.tar.gz",
                    "turing:" + job["remote"] + "/",
                    str(dest) + "/",
                ],
                check=True,
                timeout=120,
            )
            results = {}
            for lane in range(4):
                lane_status = dest / f"lane{lane}-status.json"
                record = (
                    json.loads(lane_status.read_text())
                    if lane_status.exists()
                    else {"status": "missing"}
                )
                measured = dest / f"lane{lane}/research-result.json"
                if measured.exists():
                    record["result"] = json.loads(measured.read_text())
                results[str(lane)] = record
            (dest / "collected-summary.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            status[key]["collected"] = True
            done.add(key)
        previous = (
            json.loads((BASE / "collector-status.json").read_text())
            if (BASE / "collector-status.json").exists()
            else {}
        )
        previous.update(status)
        (BASE / "collector-status.json").write_text(
            json.dumps(previous, indent=2) + "\n"
        )
        time.sleep(30)


if __name__ == "__main__":
    main()
