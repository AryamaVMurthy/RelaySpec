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
    parser.add_argument("--train-records", type=int, default=2048)
    parser.add_argument("--validation-records", type=int, default=1024)
    parser.add_argument("--training-config", type=Path)
    args = parser.parse_args()
    pilot = json.loads(args.pilot_gate.read_text())
    if pilot["status"] != "pass":
        raise ValueError("full extraction requires the successful end-to-end pilot")
    training_config = args.training_config or Path(
        "configs/protocol_active/train_dflash_qwen3_8b_relative_4gpu.yaml"
    )
    if (
        args.training_config
        and pilot.get("training_config_sha256")
        != hashlib.sha256(training_config.read_bytes()).hexdigest()
    ):
        raise ValueError("extraction pilot did not test this training configuration")
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
    config = yaml.safe_load(training_config.read_text())
    pilot_index_path = Path(pilot["feature_cache"]) / "cache-index.json"
    pilot_index = json.loads(pilot_index_path.read_text())
    if (
        pilot.get("feature_cache_index_sha256")
        and hashlib.sha256(pilot_index_path.read_bytes()).hexdigest()
        != pilot["feature_cache_index_sha256"]
    ):
        raise ValueError("pilot feature-cache index changed")
    entry = pilot_index["entries"][0]
    # Conservative float32 allowance, even though current frozen models use BF16.
    per_record = (
        config["relay_training"]["max_length"]
        * (entry["input_width"] + entry["output_width"])
        * 4
        + 65536
    )
    required = (args.train_records + args.validation_records) * per_record
    free = shutil.disk_usage(cache.parent).free
    if free < required:
        raise ValueError(f"cache needs {required} free bytes, found {free}")
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
