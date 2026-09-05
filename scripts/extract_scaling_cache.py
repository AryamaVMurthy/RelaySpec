"""Extract the declared full fitting cache after a successful bounded pilot."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-gate", type=Path, required=True)
    parser.add_argument("--train-records", type=int, default=32768)
    parser.add_argument("--validation-records", type=int, default=1024)
    args = parser.parse_args()
    if json.loads(args.pilot_gate.read_text())["status"] != "pass":
        raise ValueError("full extraction requires the successful end-to-end pilot")
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    cache = Path(os.environ["RELAYSPEC_FEATURE_CACHE"])
    if cache.exists():
        raise ValueError("refusing to overwrite a feature cache")
    output.mkdir(parents=True, exist_ok=True)
    data = Path(os.environ["RELAYSPEC_CACHE_DIR"]) / "relayspec/scaling-data/numina-v1"
    gate = json.loads((data / "manifest-gate.json").read_text())
    for name, count in [
        ("train-32768.json", args.train_records),
        ("validation.json", args.validation_records),
    ]:
        payload = (data / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != gate["files"][name]["sha256"]:
            raise ValueError(f"manifest hash mismatch: {name}")
        if count < 4 or count > len(json.loads(payload)["records"]):
            raise ValueError(f"invalid requested cache record count: {name}")
    # DFlash 8B raw taps plus teacher: <=192*23040*2 bytes per record,
    # with additional headroom for metadata, serialization and output norm.
    required = (args.train_records + args.validation_records) * 12_000_000
    free = shutil.disk_usage(cache.parent).free
    if free < required:
        raise ValueError(f"cache needs {required} free bytes, found {free}")
    config = yaml.safe_load(
        Path(
            "configs/protocol_active/train_dflash_qwen3_8b_relative_4gpu.yaml"
        ).read_text()
    )
    config["relay_training"].update(
        manifest_path=str(data / "train-32768.json"),
        validation_manifest_path=str(data / "validation.json"),
        feature_cache={
            "train_records": args.train_records,
            "validation_records": args.validation_records,
        },
    )
    path = output / "cache-config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    started = time.perf_counter()
    subprocess.run(
        [
            os.environ["RELAYSPEC_PYTHON"],
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            "scripts/train_relay.py",
            "--config",
            str(path),
        ],
        check=True,
    )
    result = json.loads((output / "feature-cache-gate.json").read_text())
    expected = {"train": args.train_records, "validation": args.validation_records}
    if result["status"] != "pass" or result["counts"] != expected:
        raise ValueError("full extraction failed its declared coverage gate")
    (output / "extraction-complete.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "counts": expected,
                "wall_seconds_including_model_load": time.perf_counter() - started,
                "pilot_gate_sha256": hashlib.sha256(
                    args.pilot_gate.read_bytes()
                ).hexdigest(),
                "cache_index_sha256": result["cache_index_sha256"],
                "total_bytes": result["total_bytes"],
                "cache_root": str(cache),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
