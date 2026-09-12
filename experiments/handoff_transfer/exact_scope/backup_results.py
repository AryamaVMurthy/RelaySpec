"""Copy measured results locally, then refresh the active campaign until Slurm ends.

No deletion, model weights, or feature caches. Live JSONL files are provisional.
Run on the local workstation, not the cluster login node.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

HOST = "aryama.murthy@turing.iiit.ac.in"
ROOT = "/scratch/node07/aryama.murthy/handoff-transfer-20260911"
CONTROL = "/home/aryama.murthy/relayspec-auf-20260911"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST]
CAMPAIGN = "single128e1-20260912"


def sync(source, destination, filters=()):
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "rsync", "-az", "--timeout=90", "--exclude=__pycache__",
        "--exclude=vllm-cache", "--exclude=inductor-cache",
        *filters, "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15",
        f"{HOST}:{source}/", str(destination) + "/",
    ], check=True)


def snapshot(destination, queue):
    files = []
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "backup.log", "watcher.pid"}:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        files.append({"path": str(path.relative_to(destination)),
                      "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    record = {
        "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": HOST, "slurm_queue": queue,
        "scope": "Raw evaluation outputs, reports, source snapshots, configs and GPU/job logs",
        "excluded": ["Model weights", "Feature caches", "Full training datasets"],
        "live_files_are_provisional": bool(queue.strip()),
        "file_count": len(files), "total_bytes": sum(x["bytes"] for x in files),
        "files": files,
    }
    temporary = destination / "manifest.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(destination / "manifest.json")
    print(record["updated_utc"], len(files), record["total_bytes"],
          "active=" + str(bool(queue.strip())), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--campaign", default=CAMPAIGN)
    parser.add_argument("--jobs", nargs='+', default=['32473','32478'])
    parser.add_argument("--active-only", action='store_true')
    args = parser.parse_args()
    for name in ([] if args.active_only else ["exact32e1b8-results", "gpu-matched-references",
                 "native-comparison-e1b8", "transformers-e1b8"]):
        sync(f"{ROOT}/{name}", args.destination / name)
    if not args.active_only:
        sync(f"{ROOT}/matrix32e1b8", args.destination / "training-metadata",
             ["--include=*/", "--include=*.json", "--include=*.jsonl",
              "--include=*.csv", "--include=*.log", "--exclude=*"])
    while True:
        # Query before copying: if jobs disappear during transfer, the next pass
        # still performs a complete final refresh after they have stopped.
        queue = subprocess.check_output(
            SSH + ["squeue -h -u aryama.murthy -o '%i %T %j'"], text=True)
        queue = "\n".join(line for line in queue.splitlines()
                          if line.split()[0].split("_")[0] in set(args.jobs))
        sync(f"{ROOT}/{args.campaign}", args.destination / args.campaign)
        sync(f"{CONTROL}/outputs", args.destination / "control-outputs")
        sync(f"{CONTROL}/logs", args.destination / "job-logs")
        snapshot(args.destination, queue)
        if not args.watch or not queue.strip():
            break
        time.sleep(45)


if __name__ == "__main__":
    main()
