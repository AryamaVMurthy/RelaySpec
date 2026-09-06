"""Audit precision-conditioned output agreement on identical question shards."""

import hashlib
import json
from pathlib import Path

import yaml

from relayspec.ar_paper_evidence import summarize

root = Path("reports/autoresearch-20260907/run-28459")
inputs = {}
results = []
all_rows = {}
for precision, lanes in [("bfloat16", [0, 2]), ("float32", [1, 3])]:
    rows = []
    for lane in lanes:
        status_path = root / f"lane{lane}-status.json"
        if json.loads(status_path.read_text())["status"] != "pass":
            raise ValueError("Incomplete precision diagnostic")
        config_path = root / f"lane{lane}/config.yaml"
        config = yaml.safe_load(config_path.read_text())
        if config["benchmark"]["precision"] != precision:
            raise ValueError("Runtime precision mismatch")
        manifest_path = Path(config["benchmark"]["manifest_path"])
        expected = {r["problem_id"] for r in json.loads(manifest_path.read_text())["records"]}
        path = root / f"lane{lane}/benchmark-rank0.jsonl"
        shard = [json.loads(x) for x in path.read_text().splitlines()]
        if {r["problem_id"] for r in shard} != expected or len(expected) != 8:
            raise ValueError("Precision shard differs from declaration")
        for r in shard:
            if r["method"] == "relay_base" and r["mapper_checkpoint_sha256"] != "d669e6334a5cc018fb6a70febda5e710cb8033152ecd1152d5bc219b6ef7a643":
                raise ValueError("Mapper checkpoint changed")
        rows.extend(shard)
        for p in [path, config_path, manifest_path, status_path]:
            inputs[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    summary = summarize(rows, reference="native_ar")
    if summary["requests"] != 16:
        raise ValueError("Missing paired requests")
    all_rows[precision] = {(r["problem_id"], r["method"]): r for r in rows}
    results.append(dict(precision=precision, **summary))
if all_rows["bfloat16"].keys() != all_rows["float32"].keys():
    raise ValueError("Precisions used different request/method pairs")
cross_precision = {
    method: sum(r["output_hash"] == all_rows["float32"][key]["output_hash"]
                for key, r in all_rows["bfloat16"].items() if key[1] == method)
    for method in ["native_ar", "relay_base"]
}
Path("reports/autoresearch-20260907/precision-summary.json").write_text(json.dumps(
    dict(input_sha256=inputs, results=results, cross_precision_output_matches=cross_precision,
         scope="Same sixteen exposed questions and fitted checkpoint. All model/mapper "
               "execution changes precision together; FP32 disables TF32. Kernel dispatch "
               "can also change. Tests within-precision AR agreement, not a proof of "
               "agreement for all prompts or attribution to target precision alone."), indent=2) + "\n")
for result in results:
    m = result["methods"]["relay_base"]
    print(result["precision"], "matches", m["token_matches"], "of", result["requests"],
          "throughput / AR", m["throughput_ratio"], m["throughput_ci95"])
print("Cross-precision agreement", cross_precision)
