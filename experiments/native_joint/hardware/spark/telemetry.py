"""Bounded, read-only GPU and UMA telemetry; no optional Python dependencies."""
import csv
import json
import os
from pathlib import Path
import subprocess
import threading
import time

FIELDS = ["uuid", "name", "utilization.gpu", "utilization.memory", "memory.total",
          "memory.used", "memory.free", "power.draw", "power.limit", "temperature.gpu",
          "clocks.current.sm", "clocks.current.memory", "pstate"]


def capture(command):
    try:
        r = subprocess.run(command, capture_output=True, text=True, timeout=5)
        return {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}


def sample():
    begin = time.monotonic()
    raw = capture(["nvidia-smi", "--query-gpu=" + ",".join(FIELDS),
                   "--format=csv,noheader,nounits"])
    devices = []
    if raw.get("returncode") == 0:
        for row in csv.reader(raw["stdout"].splitlines()):
            if len(row) != len(FIELDS):
                raise ValueError("Unexpected nvidia-smi telemetry columns")
            devices.append(dict(zip(FIELDS, (v.strip() for v in row))))
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "MemFree", "Cached", "SwapTotal", "SwapFree"}:
            memory[key + "_bytes"] = int(value.split()[0]) * 1024
    rss = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value = line.split(":", 1)
            rss[key + "_bytes"] = int(value.split()[0]) * 1024
    end = time.monotonic()
    return {"monotonic": (begin + end) / 2, "wall_time": time.time(),
            "query_duration_seconds": end - begin, "devices": devices,
            "query_failure": raw if raw.get("returncode") != 0 else None,
            "system_memory": memory, "process_memory": rss,
            "loadavg": list(os.getloadavg()),
            "cpu_stat": Path("/proc/stat").read_text().splitlines()[0]}


class Monitor:
    def __init__(self, path, interval=1.0):
        self.path, self.interval = Path(path), interval
        self.stop_event = threading.Event()
        self.error = None

    def _run(self):
        try:
            with self.path.open("x") as stream:
                while not self.stop_event.is_set():
                    start = time.monotonic()
                    stream.write(json.dumps(sample()) + "\n")
                    stream.flush()
                    self.stop_event.wait(max(0, self.interval - (time.monotonic() - start)))
        except Exception as e:
            self.error = repr(e)

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=7)
        if self.thread.is_alive():
            raise RuntimeError("Telemetry collector did not stop")
        if self.error:
            raise RuntimeError("Telemetry collector failed: " + self.error)
