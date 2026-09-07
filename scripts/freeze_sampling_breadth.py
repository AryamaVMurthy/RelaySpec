"""Freeze a separate sampling breadth and RNG-seed replication cohort."""

import copy
import hashlib
import json
import random
import re
from pathlib import Path

import yaml

root = Path("configs/autoresearch/20260907")
out = root / "sampling-breadth-v1"
assert not out.exists()
assert all(not (root / f"wave{i}.json").exists() for i in range(66, 74))
manifest = Path("configs/eval_manifest_full_v4.json")
records = [
    r for r in json.loads(manifest.read_text())["records"] if r["benchmark"] == "gsm8k"
]
reserve = Path("configs/submission/confirmation-gsm8k-20260906/gsm8k-reserve-256.json")
excluded = {r["problem_id"] for r in json.loads(reserve.read_text())["records"]}
old_cfg = yaml.safe_load((root / "sampling-full-answer-lane0.yaml").read_text())
old = [
    r
    for r in json.loads(Path(old_cfg["benchmark"]["manifest_path"]).read_text())[
        "records"
    ]
    if r["benchmark"] == "gsm8k"
]
random.Random(old_cfg["seed"]).shuffle(old)
old_ids = {r["problem_id"] for r in old[:8]}
excluded |= old_ids
records = [r for r in records if r["problem_id"] not in excluded]
random.Random(1729).shuffle(records)
records = records[:32]
ids = {r["problem_id"] for r in records}
assert len(ids) == 32 and not ids & excluded
pattern = re.compile(
    rb'"problem_id"\s*:\s*"('
    + b"|".join(re.escape(x.encode()) for x in sorted(ids))
    + rb')"'
)
exposure = {p: [] for p in sorted(ids)}
files = 0
for base in [Path("reports"), Path("/home/aryamavmurthy/work/RelaySpec/reports")]:
    for p in sorted(base.rglob("*.jsonl")):
        files += 1
        for match in {m.decode() for m in pattern.findall(p.read_bytes())}:
            exposure[match].append(str(p))
assert all(exposure.values()), "This protocol declares an exposed development cohort"
out.mkdir()
(out / "manifest32.json").write_text(json.dumps({"records": records}, indent=2) + "\n")
(out / "exposure.json").write_text(json.dumps(exposure, indent=2) + "\n")
for i in range(4):
    (out / f"shard{i}.json").write_text(
        json.dumps({"records": records[8 * i : 8 * (i + 1)]}, indent=2) + "\n"
    )
template = json.loads((root / "wave64.json").read_text())
protocol = dict(
    status="frozen-before-decoding",
    records=32,
    sampling_seed_bases=[1730, 1731],
    families=["dflash", "eagle3"],
    temperatures=[0.6, 1.0],
    max_new_tokens=2048,
    source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    manifest_sha256=hashlib.sha256((out / "manifest32.json").read_bytes()).hexdigest(),
    exposure_files_scanned=files,
    exposure="All32 questions have prior local evaluation records. Exact IDs are disjoint from the entire256-question reserve and the separate eight-question sampling study. No fresh-question confirmation claim.",
    checkpoints={
        p: sha for s in template["lanes"] for p, sha in s["checkpoint_sha256"].items()
    },
    analysis="Fixed32 questions x2 sampling seeds per family/temperature. Pair methods within worker. Bootstrap question clusters jointly across both seeds with10000 draws seed1729. Report token throughput, request time, output length, scored-answer counts and uncertainty. No quality noninferiority, distributional exactness or joint significance claim. Do not pool the separate eight-question study.",
    execution="Eight four-GPU waves, each lane handles eight questions and three methods with a540-second timeout. All declared cells required for complete aggregate. Failures remain recorded. No outcome-dependent sample extension or candidate tuning.",
    waves=[],
)
for seed_index, seed in enumerate([1730, 1731]):
    for shard in range(4):
        number = 66 + seed_index * 4 + shard
        wave = copy.deepcopy(template)
        for lane, s in enumerate(wave["lanes"]):
            cfg = yaml.safe_load(Path(s["config"]).read_text())
            cfg["benchmark"]["manifest_path"] = str(out / f"shard{shard}.json")
            cfg["benchmark"]["sampling_seed_base"] = seed
            name = out / f"wave{number}-lane{lane}.yaml"
            name.write_text(yaml.safe_dump(cfg, sort_keys=False))
            s["config"] = str(name)
            s["hypothesis"] = (
                "Does sampled RelaySpec applicability repeat across32 fixed development questions and two independent sampling seeds in both drafter families?"
            )
            s["scope"] = (
                "Fixed sampling-breadth-v1 shard with8 questions,2 temperatures and2 families across the four lanes. One of two predeclared sampling_seed_bases. Frozen512-record mappers with native/AR controls,2048token cap. All32 questions previously exposed, distinct from initial8 sampling questions and entire reserve. All lanes540seconds."
            )
        (root / f"wave{number}.json").write_text(json.dumps(wave, indent=2) + "\n")
        protocol["waves"].append(
            dict(wave=number, sampling_seed_base=seed, shard=shard)
        )
(out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
print(
    json.dumps(
        dict(
            records=32,
            files_scanned=files,
            all_previously_exposed=True,
            reserve_overlap=0,
            initial_sampling_overlap=0,
            waves=list(range(66, 74)),
        )
    )
)
