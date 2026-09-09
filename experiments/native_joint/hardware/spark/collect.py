"""Bounded read-only remote status monitor and completed-result collector."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--socket", required=True)
    p.add_argument("--remote", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    ssh = ["ssh", "-S", a.socket, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", a.host]
    previous = None
    deadline = time.monotonic() + 10*3600
    while time.monotonic() < deadline:
        try:
            raw = subprocess.check_output(ssh + ["cat " + shlex.quote(a.remote + "/status.json")], text=True, timeout=30)
            status = json.loads(raw)
            (a.output / "remote-status.json").write_text(json.dumps(status, indent=2))
            stage = status["stage"]
            if stage != previous:
                print(json.dumps(status), flush=True)
                previous = stage
            if stage in {"complete", "failed"}:
                subprocess.run(["rsync", "-a", "--partial", "-e", shlex.join(ssh[:-1]),
                                a.host + ":" + a.remote + "/", str(a.output / "results") + "/"], check=True, timeout=1800)
                (a.output / "collected.json").write_text(json.dumps({"stage": stage, "remote": a.remote}))
                return
        except (subprocess.SubprocessError, json.JSONDecodeError) as error:
            print(json.dumps({"collector_error": str(error)}), flush=True)
        time.sleep(60)
    raise TimeoutError("Remote collection exceeded ten hours; remote job was not terminated")


if __name__ == "__main__":
    main()
