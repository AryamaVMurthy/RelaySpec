"""Freeze the existing dense small-data comparison before confirmation generation."""

import copy
import hashlib
import json
import random
from pathlib import Path
import yaml


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    base = Path("configs/submission/confirmation-gsm8k-20260906")
    out = base / "frozen-dense-v1"
    out.mkdir(exist_ok=True)
    reserved = json.loads((base / "gsm8k-confirmation-256.json").read_text())["records"]
    all_ids = {
        r["problem_id"]
        for n in ["gsm8k-confirmation-256.json", "gsm8k-reserve-256.json"]
        for r in json.loads((base / n).read_text())["records"]
    }
    source = Path("configs/eval_manifest_full_v4.json")
    exposed = [
        r
        for r in json.loads(source.read_text())["records"]
        if r["benchmark"] == "gsm8k"
    ]
    random.Random(1729).shuffle(exposed)
    exposed = exposed[:8]
    assert len(exposed) == 8 and not all_ids.intersection(
        r["problem_id"] for r in exposed
    )
    for family, config_path, provpath in [
        (
            "dflash",
            "configs/submission/scaling/campaign-small-data-quality-development.yaml",
            "configs/submission/scaling/small-data-quality.provenance.json",
        ),
        (
            "eagle3",
            "configs/submission/scaling/eagle3-small-quality-v1/campaign-full0.yaml",
            "configs/submission/scaling/eagle3-small-quality-v1/protocol.json",
        ),
    ]:
        cfg = yaml.safe_load(Path(config_path).read_text())
        prov = json.loads(Path(provpath).read_text())
        variants = {
            k: {**prov["variants"][k], "checkpoint": v}
            for k, v in cfg["relay_probe"]["variants"].items()
            if "dense_n" in k
        }
        assert len(variants) == 2
        cfg["relay_probe"]["variants"] = {
            k: v["checkpoint"] for k, v in variants.items()
        }
        cfg["benchmark"]["methods"] = [
            m for m in cfg["benchmark"]["methods"] if not m.startswith("relay_")
        ] + list(variants)
        cfg["benchmark"]["benchmarks"] = ["gsm8k"]
        stages = {}
        for stage, records in [("pilot", exposed), ("full", reserved)]:
            mp = out / f"{family}-{stage}-manifest.json"
            mp.write_text(
                json.dumps(
                    {
                        "records": records,
                        "scope": "Exposed pilot"
                        if stage == "pilot"
                        else "Frozen primary confirmation membership",
                    },
                    indent=2,
                )
                + "\n"
            )
            c = copy.deepcopy(cfg)
            c["run_name"] = f"confirmation-{family}-{stage}"
            c["benchmark"].update(manifest_path=str(mp), max_prompts=len(records))
            cp = out / f"{family}-{stage}.yaml"
            cp.write_text(yaml.safe_dump(c, sort_keys=False))
            stages[stage] = {
                "config": str(cp),
                "config_sha256": digest(cp),
                "manifest": str(mp),
                "manifest_sha256": digest(mp),
                "requests": len(records),
                "problem_ids": [r["problem_id"] for r in records],
            }
        p = {
            "status": "frozen_before_confirmation_generation",
            "family": family,
            "benchmark": "gsm8k",
            "input_sha256": {
                str(p): digest(p)
                for p in [
                    config_path,
                    provpath,
                    base / "protocol.json",
                    source,
                    Path(__file__),
                    "reports/mapper-scaling-20260905/resumption-preflight.json",
                ]
            },
            "methods": cfg["benchmark"]["methods"],
            "variants": variants,
            "stages": stages,
            "dense_reference": next(k for k in variants if "n2048" in k),
            "candidate": next(k for k in variants if "n512" in k),
            "output_cap": 2048,
            "selection": "Dense N512 seed1729 step8192 versus dense N2048 seed1729 step8192, selected using completed development evidence. No retraining or changes based on confirmation.",
            "primary_role": "Primary DFlash test"
            if family == "dflash"
            else "Separate cross-family replication on the same questions; not independent task replication",
            "speed_metric": "Aggregate generated tokens / end-to-end request seconds including prompt processing, aligned with manuscript throughput. Pre-generation clarification of ambiguous decode-second wording in parent protocol.",
            "speed_threshold": 0.95,
            "quality_margin": 0.01,
            "quality_interval": "Conservative paired Clopper-Pearson construction, scipy1.16.1. N256 cannot establish one-point noninferiority even with all pairs concordant. Preserve inconclusive outcomes.",
            "execution": "Pilot8 previously exposed GSM8K questions per family under540 seconds. Full256 primary questions only after pilot audit. Four GPUs total. Reserve256 unused.",
            "scope": "Frozen-checkpoint confirmation relative to scanned local history and lexical filtering. No global exposure, pretrained-model contamination or semantic independence guarantee. Retain all outcomes; no tuning or optional reserve extension.",
        }
        (out / f"{family}-protocol.json").write_text(json.dumps(p, indent=2) + "\n")
        print(family, len(variants), len(reserved))


if __name__ == "__main__":
    main()
