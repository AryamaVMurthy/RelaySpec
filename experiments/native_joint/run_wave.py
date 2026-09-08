import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

specs = json.loads(Path(os.environ["WAVE"]).read_text())["lanes"]
devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
if len(devices) != 4 or len(specs) != 4:
    raise RuntimeError("Four allocated GPUs and four lanes required")
output = Path("outputs")/os.environ["SLURM_JOB_ID"]
output.mkdir(parents=True, exist_ok=False)


def run(i):
    config = output/f"lane{i}.json"
    config.write_text(json.dumps(specs[i], indent=2))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[i])
    limit = int(specs[i].get("timeout_seconds", 540))
    if not 0 < limit <= (540 if i < 2 else 900):
        raise ValueError("Two lanes must remain within ten minutes")
    start = time.perf_counter()
    with (output/f"lane{i}.log").open("w") as log:
        lane_script = specs[i].get("lane_script", os.environ.get("LANE_SCRIPT", "run_lane.py"))
        result = subprocess.run(["timeout", "--signal=TERM", "--kill-after=10s", f"{limit}s", sys.executable, lane_script, "--config", str(config), "--output", str(output/f"lane{i}")], env=env, stdout=log, stderr=subprocess.STDOUT)
    status = {"lane": i, "exit_code": result.returncode, "seconds": time.perf_counter()-start}
    (output/f"lane{i}-status.json").write_text(json.dumps(status, indent=2))
    return status


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(run, range(4)))
(output/"wave-result.json").write_text(json.dumps(results, indent=2))
raise SystemExit(any(r["exit_code"] for r in results))
