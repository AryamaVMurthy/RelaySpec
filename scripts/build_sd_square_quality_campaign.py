"""Freeze the best tested SD-square epoch before eight longer quality requests."""

import argparse
import json
from pathlib import Path

from build_sd_square_campaign import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    registry_path = Path(
        "reports/external-baselines-20260906/sd-square-epoch-decoding.json"
    )
    registry = json.loads(registry_path.read_text())
    if (
        registry["status"] != "complete"
        or registry["comparisons"]["native_ar"]["requests"] != 16
    ):
        raise ValueError("SD-square epoch comparison is not complete")
    name = max(
        (n for n, v in registry["variants"].items() if v["kind"] == "steering"),
        key=lambda n: registry["comparisons"]["native_ar"]["methods"][n][
            "tokens_per_second"
        ],
    )
    variant = dict(registry["variants"][name])
    identity = registry["steering_fingerprints"][name]
    if identity["training_sha256"] != variant["trainable_sha256"]:
        raise ValueError("SD-square selected epoch lacks matching training provenance")
    variant["inference_sha256"] = identity["inference_sha256"]
    if (
        args.check_files
        and digest(Path(variant["checkpoint"])) != variant["checkpoint_sha256"]
    ):
        raise ValueError("SD-square selected checkpoint file changed")
    source_path = Path("configs/submission/baselines/sd-square-compatibility.json")
    protocol_path = Path(
        "configs/submission/baselines/external-verification-protocol.json"
    )
    protocol = json.loads(protocol_path.read_text())
    if protocol["development"]["quality_pilot_requests"] != 8:
        raise ValueError("SD-square quality pilot request declaration changed")
    result = {
        "source_config": str(source_path),
        "source_config_sha256": digest(source_path),
        "verification_protocol": str(protocol_path),
        "verification_protocol_sha256": digest(protocol_path),
        "prerequisites": {str(registry_path): digest(registry_path)},
        "phase": "quality_pilot",
        "selected_epoch_method": name,
        "selection_rule": "highest throughput point among all four predeclared epoch settings",
        "seed": 1729,
        "manifest_path": "configs/eval_manifest.json",
        "requests": 4,
        "request_offset": 4 * args.shard_index,
        "total_development_requests": 8,
        "max_new_tokens": protocol["development"]["quality_token_cap"],
        "observer_pilot_tokens": 64,
        "warmup_tokens": 16,
        "inference_precision": registry["inference_precision"],
        "variants": {"native_ar": {"kind": "ar"}, "sd2_selected": variant},
        "fitting_input_sha256": registry["fitting_input_sha256"],
        "original_exact_ar_gates": registry["original_exact_ar_gates"],
        "scope": "Selected SD-square epoch frozen before eight exposed development requests at2048-token cap. "
        "Two disjoint four-request pilot shards and correct runtime-local AR. Public FP16 target/BF16 "
        "drafter and steering with BF16 autocast. Training and inference identities must match prior decoding. "
        "Strict actual-verifier checks and immediate/deferred observer equivalence. Full request time "
        "includes GPU observation and excludes CPU materialization/detokenization. Capped answer scoring "
        "is separate from timing. This pilot does not establish final quality noninferiority, untouched "
        "confirmation, isolated memory or an algorithm-only cross-runtime ranking.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
