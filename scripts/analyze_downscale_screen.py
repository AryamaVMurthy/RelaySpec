"""Audit and score downward-retargeting outputs against paired AR measurements."""

import hashlib
import json
import os
import random
from pathlib import Path

import yaml
from aggregate_results import build_math_scorer

from relayspec.ar_paper_evidence import summarize
from relayspec.paired_accuracy import paired_accuracy_interval

repo = Path.cwd()
run = repo / "reports/autoresearch-20260907/run-28458"
config_path = repo / "configs/autoresearch/20260907/downscale-dflash-06b-trained-screen.yaml"
expected = yaml.safe_load(config_path.read_text())
actual = yaml.safe_load((run / "config.yaml").read_text())
if actual != expected:
    raise ValueError("Decoding configuration differs from declaration")
gate = json.loads((run / "campaign-gate.json").read_text())
if gate["status"] != "pass" or gate["duplicate_map_equivalence"]["status"] != "pass":
    raise ValueError("Incomplete or inequivalent duplicate-map evaluation")
fit_gate_path = repo / "reports/autoresearch-20260907/run-28457/batch-gate.json"
fit_gate = json.loads(fit_gate_path.read_text())
if fit_gate["status"] != "pass":
    raise ValueError("Full fitting did not pass")
manifest_path = repo / expected["benchmark"]["manifest_path"]
candidates = [r for r in json.loads(manifest_path.read_text())["records"]
              if r["benchmark"] in expected["benchmark"]["benchmarks"]]
random.Random(expected["seed"]).shuffle(candidates)
selected = candidates[:expected["benchmark"]["max_prompts"]]
ids = {r["problem_id"] for r in selected}
methods = expected["benchmark"]["methods"]
paths = [run / f"benchmark-rank{i}.jsonl" for i in range(4)]
rows = [json.loads(x) for p in paths for x in p.read_text().splitlines()]
keys = {(r["problem_id"], r["method"], r["repetition"]) for r in rows}
if len(keys) != len(rows) or keys != {(p, m, 0) for p in ids for m in methods}:
    raise ValueError("Incomplete, duplicate, or unexpected measurements")
for r in rows:
    checkpoint = expected["relay_probe"]["variants"].get(r["method"])
    if checkpoint and r["mapper_checkpoint_sha256"] != fit_gate["checkpoint_sha256"][checkpoint]:
        raise ValueError("Decoded checkpoint differs from completed fit")
summary = summarize(rows, reference="native_ar")
inputs = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in [config_path, fit_gate_path, manifest_path, *paths]}

scoring_repo = Path("/home/aryamavmurthy/work/RelaySpec")
os.chdir(scoring_repo)
provenance_path = Path("reports/ar-revision-20260905/scorer-provenance.json")
provenance = json.loads(provenance_path.read_text())
for path, sha in provenance["sha256"].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
        raise ValueError("Pinned scorer changed")
scorer, scorer_name = build_math_scorer(Path("vendor/qwen-math/evaluation"))
cache = {}
for r in rows:
    key = (r["completion"], str(r["reference_answer"]), r["benchmark"])
    if key not in cache:
        cache[key] = scorer(*key)
    r.update(cache[key])
lookup = {(r["problem_id"], r["method"]): r for r in rows}
quality, profiles = {}, {}
for method in methods:
    mr = [r for r in rows if r["method"] == method]
    quality[method] = dict(correct=sum(r["correct"] for r in mr),
                           capped=sum(r["output_tokens"] >= 512 for r in mr),
                           paired_accuracy=paired_accuracy_interval(
                               [lookup[p, method]["correct"] for p in sorted(ids)],
                               [lookup[p, "native_ar"]["correct"] for p in sorted(ids)]))
    seconds = sum(r["request_seconds"] for r in mr)
    regions = {}
    for r in mr:
        for name, milliseconds in r["profile_regions_ms"].items():
            regions[name] = regions.get(name, 0.0) + milliseconds / 1000
    profiles[method] = {name: value / seconds for name, value in regions.items()}
(run / "downscale-scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
(repo / "reports/autoresearch-20260907/downscale-summary.json").write_text(json.dumps(
    dict(input_sha256=inputs, summary=summary, quality=quality, profile_fractions=profiles,
         scorer=scorer_name, scorer_provenance=provenance,
         scope="Sixteen exposed development GSM8K requests; 512-token cap. "
               "Paired AR reference, fixed512-record fits,8192updates. Duplicate maps "
               "are controls, not independent fitting runs. No fresh-confirmation claim."), indent=2) + "\n")
for method in methods:
    m = summary["methods"][method]
    print(method, round(m["tokens_per_second"], 2), m["throughput_ratio"],
          m["throughput_ci95"], "progress", round(m["progress_per_cycle"], 3),
          "correct", quality[method]["correct"], "capped", quality[method]["capped"],
          "matching AR outputs", m["token_matches"])
