"""Declare equal-count composition fits, with an exact-cache pilot per arm."""

import argparse
import hashlib
import json
from pathlib import Path

from relayspec.scaling_matrix import trial, validate_trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate_path = Path(
        "reports/mapper-scaling-20260905/composition-data/manifest-gate.json"
    )
    gate = json.loads(gate_path.read_text())
    if (
        gate["status"] != "pass"
        or gate.get("validation_order") != "alternating_math_general_instruction_v2"
    ):
        raise ValueError(
            "composition requires the checked interleaved validation manifests"
        )
    if any(info["records"] != 2048 for info in gate["files"].values()):
        raise ValueError("composition pool must remain at 2048 records")
    args.output.mkdir(parents=True, exist_ok=False)
    arms = {}
    for arm in ["math", "mixed"]:
        cells, pilots = [], []
        for kind, width, seed in [
            ("dense", None, 1729),
            ("factorized", 512, 1729),
            ("mlp", 512, 1729),
            ("dense", None, 1730),
        ]:
            cell = trial(
                kind, width, 2048, seed=seed, study="matched_count_composition"
            )
            cell.update(
                name=f"composition-{arm}-{cell['name']}",
                checkpoint_steps=[2048, 8192],
                validation_records=2048,
                report_domain_diagnostics=True,
                validation_required_domains=["general_instruction", "math"],
                cache_backend="buffer",
                device_cache=False,
            )
            validate_trial(cell)
            cells.append(cell)
            pilot = {
                **cell,
                "name": "pilot-" + cell["name"],
                "steps": 16,
                "checkpoint_steps": [16],
                "budget_panels": {},
            }
            validate_trial(pilot)
            pilots.append(pilot)
        for suffix, trials in [("fits", cells), ("pilot", pilots)]:
            (args.output / f"{arm}-{suffix}.json").write_text(
                json.dumps({"trials": trials}, indent=2) + "\n"
            )
        arms[arm] = {"training_manifest": f"train-{arm}-2048.json", "fits": cells}
    (args.output / "protocol.json").write_text(
        json.dumps(
            {
                "status": "declared_before_composition_fitting",
                "large_data_scaling": "paused",
                "data_root_tag": "math-dolly-v2",
                "manifest_gate_sha256": hashlib.sha256(
                    gate_path.read_bytes()
                ).hexdigest(),
                "training_config": "configs/protocol_active/train_dflash_qwen3_8b_relative_4gpu.yaml",
                "campaign_config": "configs/submission/scaling/campaign-pilot.yaml",
                "arms": arms,
                "execution": "Each arm requires a 64-train/16-validation cache-fit-decode compatibility pilot, then 2048/2048 frozen extraction, then an exact-cache 16-update resource/decode pilot before 8192-update fits. All pilots have a 540-second process timeout and 10-minute Slurm limit. Sequential four-GPU allocations.",
                "scope": "Six primary fits plus two dense seed controls. 32768 record presentations reuse each 2048-example pool. Common validation is diagnostic only and cannot select a horizon. Compare actual non-padding training tokens, exposed development decoding, and full-answer quality separately. Not large-data scaling or untouched confirmation.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
