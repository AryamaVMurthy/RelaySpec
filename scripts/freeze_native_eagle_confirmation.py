"""Freeze one native EAGLE candidate and the next unused reserve block."""

import copy
import hashlib
import json
import re
from pathlib import Path

import yaml

root = Path("configs/autoresearch/20260907")
out = root / "native-eagle-confirmation-v1"
if out.exists():
    raise ValueError("Refusing to overwrite a frozen confirmation protocol")
reserve = Path("configs/submission/confirmation-gsm8k-20260906/gsm8k-reserve-256.json")
records = sorted(
    json.loads(reserve.read_text())["records"], key=lambda r: r["problem_id"]
)[192:256]
ids = {r["problem_id"] for r in records}
assert len(ids) == 64
pattern = re.compile(
    rb'"problem_id"\s*:\s*"('
    + b"|".join(re.escape(s.encode()) for s in sorted(ids))
    + rb')"'
)
files = []
matches = []
for base in [Path("reports"), Path("/home/aryamavmurthy/work/RelaySpec/reports")]:
    for p in sorted(base.rglob("*.jsonl")):
        files.append(str(p))
        found = {m.decode() for m in pattern.findall(p.read_bytes())}
        if found:
            matches.append({"file": str(p), "problem_ids": sorted(found)})
if matches:
    raise ValueError(f"Reserve block has recorded exposure: {matches}")
wave63 = json.loads((root / "wave63.json").read_text())["lanes"][0]
base = wave63["checkpoint"]
long_fit = wave63["controls"]["relay_inherited_u8192"]
short_fit = wave63["controls"]["relay_inherited_u128"]
svd = wave63["controls"]["relay_svd1536"]
hashes = {p: wave63["checkpoint_sha256"][p] for p in [base, long_fit, short_fit, svd]}
protocol = dict(
    status="frozen-before-decoding",
    records=64,
    reserve_sha256=hashlib.sha256(reserve.read_bytes()).hexdigest(),
    reserve_start=192,
    reserve_stop=256,
    primary_candidate="relay_two",
    reference="native_target_eagle3",
    primary_analysis="Token-sum/time-sum throughput ratio with10000 paired question bootstrap replicates seed1729. Primary retention criterion: lower two-sided95% interval >=0.95. Fixed64 questions with no optional extension.",
    secondary="128-update512-record fit, rank1536 SVD, and full repacked projection are descriptive controls. No alternative candidate promotion if primary fails.",
    candidate_selection="First512 calibration records and8192 updates, inherited native columns for layers25/33, seed1729. Selected from completed development controls before this reserve block is decoded. Code retention is lower than math in development and the confirmation concerns GSM8K only.",
    checkpoints=hashes,
    exposure_scan=dict(
        jsonl_files_scanned=len(files),
        matches=matches,
        scope="Both local repository report trees scanned for exact problem_id before decoding. Not a global/pretraining or semantic independence guarantee.",
    ),
    quality="Pinned GSM8K scorer on all saved outputs. Conservative paired accuracy interval and cap counts retained. No1pp noninferiority claim.",
    execution="Four16-question singleGPU shards,540second cap each. All shards required, no partial-result confirmation.",
    reserve_accounting="Prior positions0:192 already used by three separate studies. This study consumes192:256, leaving none of the original reserve unused.",
)
out.mkdir()
(out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
(out / "exposure-files.json").write_text(json.dumps(files, indent=2) + "\n")
config = yaml.safe_load((root / "native-eagle-pilot-lane0.yaml").read_text())
lanes = []
for i in range(4):
    shard = out / f"shard{i}.json"
    shard.write_text(
        json.dumps({"records": records[i * 16 : (i + 1) * 16]}, indent=2) + "\n"
    )
    cfg = copy.deepcopy(config)
    cfg["benchmark"]["manifest_path"] = str(shard)
    cfgpath = out / f"lane{i}.yaml"
    cfgpath.write_text(yaml.safe_dump(cfg, sort_keys=False))
    lanes.append(
        dict(
            family="eagle3",
            transform="existing",
            checkpoint=base,
            config=str(cfgpath),
            requests=16,
            max_new_tokens=2048,
            include_ar=False,
            include_native=True,
            candidates={
                "relay_two": long_fit,
                "relay_short": short_fit,
                "relay_svd1536": svd,
            },
            checkpoint_sha256=hashes,
            hypothesis="Frozen primary:512-record8192-update inherited native EAGLE two-layer map retains at least95% throughput on64 new reserve questions.",
            scope="Fixed reserve sorted positions192:256, native EAGLE two-layer primary frozen before decoding. All outputs and controls retained, no optional extension or tuning.",
        )
    )
(root / "wave65.json").write_text(json.dumps({"lanes": lanes}, indent=2) + "\n")
print(
    json.dumps(
        {
            "records": 64,
            "files_scanned": len(files),
            "matches": matches,
            "primary_checkpoint": long_fit,
        }
    )
)
