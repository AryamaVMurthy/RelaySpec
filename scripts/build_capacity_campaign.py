"""Require every primary fit gate before declaring paired capacity decoding."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from relayspec.mapper_campaign import campaign_references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("configs/submission/scaling/matrix-focused-v1/matrix.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("configs/submission/scaling/campaign-pilot.yaml"),
    )
    args = parser.parse_args()
    primary = json.loads(args.matrix.read_text())["primary_cells"]
    gates = {}
    for p in (args.raw_root / "reports/mapper-scaling-20260905").glob(
        "*/run-*/batch-gate.json"
    ):
        g = json.loads(p.read_text())
        if g.get("status") != "pass" or g.get("mode") != "fit":
            continue
        for name in g["trials"]:
            if name in gates:
                raise ValueError(
                    f"multiple completed fits need explicit resolution: {name}"
                )
            gates[name] = (p, g)
    config = yaml.safe_load(args.template.read_text())
    references = campaign_references(
        config["proposer"]["family"], config["benchmark"]["methods"]
    )
    config["run_name"] = "small-data-complete-capacity-decoding"
    config["generation"]["max_new_tokens"] = 256
    config["benchmark"]["max_prompts"] = 16
    variants, provenance = {}, {}
    cache_hash = None
    for t in primary:
        if t["name"] not in gates:
            raise ValueError(f"primary fit is incomplete: {t['name']}")
        p, g = gates[t["name"]]
        if (
            "campaign_config_sha256" in g
            and g["campaign_config_sha256"]
            != hashlib.sha256(args.template.read_bytes()).hexdigest()
        ):
            raise ValueError(
                "fit gate was validated with a different family campaign template"
            )
        cache_hash = cache_hash or g["feature_cache_index_sha256"]
        if cache_hash != g["feature_cache_index_sha256"]:
            raise ValueError("primary fits have different frozen features")
        paths = [
            (k, sha)
            for k, sha in g["checkpoint_sha256"].items()
            if Path(k).parent.name == t["name"] and Path(k).name == "step-008192.pt"
        ]
        if len(paths) != 1:
            raise ValueError("missing or ambiguous fixed-exposure checkpoint")
        complete = json.loads(
            (p.parent / "fitting" / t["name"] / "fit-complete.json").read_text()
        )
        comparable = {
            k: v
            for k, v in complete["trial"].items()
            if k not in {"cache_backend", "device_cache"}
        }
        declared = {
            k: v for k, v in t.items() if k not in {"cache_backend", "device_cache"}
        }
        if comparable != declared:
            raise ValueError("completed fit does not match its declared primary cell")
        expected_normalization = config["proposer"]["family"] == "dflash"
        if t.get("normalize_input", expected_normalization) != expected_normalization:
            raise ValueError(
                "declared mapper normalization differs from this family protocol"
            )
        alias = "relay_" + t["name"].replace("-s1729", "").replace("-", "_")
        variants[alias] = paths[0][0]
        provenance[alias] = {
            "trial": t,
            "checkpoint_sha256": paths[0][1],
            "fit_gate_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
    config["relay_probe"]["variants"] = variants
    config["benchmark"]["methods"] = [*references, *variants]
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "status": "declared_after_all_primary_fit_gates",
                "matrix_sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "feature_cache_index_sha256": cache_hash,
                "variants": provenance,
                "scope": f"{len(primary)} predeclared primary endpoints at8192updates, N512/2048. Paired16-request256-token development decoding; neither full-answer quality nor isolated deployment memory.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Declared {len(variants)} primary mappers with shared references")


if __name__ == "__main__":
    main()
