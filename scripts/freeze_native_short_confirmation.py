"""Freeze one short-fit native candidate and the next unused reserve block."""

import copy
import hashlib
import json
import re
from pathlib import Path

import yaml

root = Path("configs/autoresearch/20260907")
out = root / "native-short-confirmation-v1"
if out.exists():
    raise ValueError("Refusing to overwrite a frozen confirmation protocol")
reserve = Path("configs/submission/confirmation-gsm8k-20260906/gsm8k-reserve-256.json")
records = sorted(
    json.loads(reserve.read_text())["records"], key=lambda r: r["problem_id"]
)[128:192]
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
wave46 = json.loads((root / "wave46.json").read_text())["lanes"][0]
base = wave46["checkpoint"]
n16 = wave46["controls"]["relay_n16_native_columns"]
n128 = wave46["controls"]["relay_n128_native_columns"]
campaign = json.loads(
    Path(
        "reports/autoresearch-20260907/run-28533/lane0/mapper-campaign.json"
    ).read_text()
)
crop = campaign["variants"]["relay_cropped"]["checkpoint"]
hashes = {p: wave46["checkpoint_sha256"][p] for p in [base, n16, n128]}
hashes[crop] = campaign["variants"]["relay_cropped"]["sha256"]
protocol = dict(
    status="frozen-before-decoding",
    records=64,
    reserve_sha256=hashlib.sha256(reserve.read_bytes()).hexdigest(),
    reserve_start=128,
    reserve_stop=192,
    primary_candidate="relay_short16",
    reference="native_target_dflash",
    primary_analysis="Token-sum/time-sum throughput ratio with10000 paired question bootstrap replicates seed1729. Primary retention criterion: lower two-sided95% interval >=0.95. Fixed64 questions with no optional extension.",
    secondary="128-record short fit, untrained cropped native, and full repacked projection are descriptive controls. No alternative candidate promotion if primary fails.",
    candidate_selection="Original first16 calibration records and128 updates, native-column initialization, seed1729, selected from development studies before this reserve block is decoded. No changes based on the scalar-calibration study.",
    checkpoints=hashes,
    exposure_scan=dict(
        jsonl_files_scanned=len(files),
        matches=matches,
        scope="Both local repository report trees scanned for exact problem_id before decoding. Not a global/pretraining or semantic independence guarantee.",
    ),
    quality="Pinned GSM8K scorer on all saved outputs. Conservative paired accuracy interval and cap counts retained. No1pp noninferiority claim.",
    execution="Four16-question singleGPU shards,540second cap each. All shards required, no partial-result confirmation.",
    reserve_accounting="Prior positions0:128 already used by separate studies. This study consumes128:192, leaving192:256 unused.",
)
out.mkdir()
(out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
(out / "exposure-files.json").write_text(json.dumps(files, indent=2) + "\n")
config = yaml.safe_load((root / "native-gsm8k-screen.yaml").read_text())
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
            family="dflash",
            transform="existing",
            checkpoint=base,
            config=str(cfgpath),
            requests=16,
            max_new_tokens=2048,
            include_ar=False,
            include_native=True,
            candidates={
                "relay_short16": n16,
                "relay_short128": n128,
                "relay_cropped": crop,
            },
            checkpoint_sha256=hashes,
            hypothesis="Frozen primary:16-record128-update inherited native compact map retains at least95% throughput on64 new reserve questions.",
            scope="Fixed reserve sorted positions128:192, primary short16 frozen before decoding. All outputs and controls retained, no optional extension or tuning.",
        )
    )
(root / "wave50.json").write_text(json.dumps({"lanes": lanes}, indent=2) + "\n")
print(
    json.dumps(
        {
            "records": 64,
            "files_scanned": len(files),
            "matches": matches,
            "primary_checkpoint": n16,
        }
    )
)
