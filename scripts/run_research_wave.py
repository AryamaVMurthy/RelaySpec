"""Four independent GPU lanes; every experiment has a540-second execution cap."""

import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    out = Path(os.environ["RELAYSPEC_OUTPUT"])
    specs = json.loads(Path(os.environ["RESEARCH_WAVE"]).read_text())["lanes"]
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(visible) != 4 or len(specs) != 4:
        raise ValueError("Exactly four allocated GPU lanes required")

    def run(i):
        spec = specs[i]
        path = out / f"lane{i}-spec.json"
        path.write_text(json.dumps(spec, indent=2) + "\n")
        family = spec["family"]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": visible[i]}
        env["PYTHONPATH"] = (
            str(Path("src").resolve())
            + ":"
            + (
                os.environ["DFLASH_SOURCE"]
                if family == "dflash"
                else os.environ["DEEPSPEC_PYTHON_OVERLAY"]
                + ":"
                + os.environ["DEEPSPEC_SOURCE"]
            )
        )
        start = time.monotonic()
        with (out / f"lane{i}.log").open("w") as log:
            result = subprocess.run(
                [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=10s",
                    "540s",
                    os.environ["RELAYSPEC_PYTHON"],
                    "scripts/run_research_lane.py",
                    "--spec",
                    str(path),
                    "--output",
                    str(out / f"lane{i}"),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        record = {
            "lane": i,
            "exit_code": result.returncode,
            "elapsed_seconds": time.monotonic() - start,
            "status": "pass"
            if result.returncode == 0
            else "timeout"
            if result.returncode == 124
            else "failed",
        }
        (out / f"lane{i}-status.json").write_text(json.dumps(record, indent=2) + "\n")
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(run, range(4)))
    (out / "wave-result.json").write_text(
        json.dumps({"lanes": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
