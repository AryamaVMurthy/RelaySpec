"""Cancel only explicitly registered jobs at the user-defined GPU cutoff."""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=Path, required=True)
    args = parser.parse_args()
    while True:
        budget = json.loads(args.budget.read_text())
        cutoff = datetime.fromisoformat(budget["gpu_stop_utc"]).timestamp()
        remaining = cutoff - time.time()
        if remaining > 0:
            time.sleep(min(20, remaining))
            continue
        ids = sorted({int(value) for value in budget["job_ids"]})
        if not ids or any(value <= 0 for value in ids):
            raise ValueError("deadline requires explicit positive job IDs")
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "turing",
                "scancel " + " ".join(map(str, ids)),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        record = {
            "checked_utc": datetime.now(timezone.utc).isoformat(),
            "job_ids": ids,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        with (args.budget.parent / "deadline-actions.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if result.returncode == 0:
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
