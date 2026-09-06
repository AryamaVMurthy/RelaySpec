"""Audit precision-conditioned output agreement on identical question shards."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from relayspec.ar_paper_evidence import summarize

parser = argparse.ArgumentParser()
parser.add_argument("--job", type=int, default=28459)
parser.add_argument("--wave", type=int, default=25)
parser.add_argument("--output", default="precision-summary.json")
args = parser.parse_args()
root = Path(f"reports/autoresearch-20260907/run-{args.job}")
specs = json.loads(Path(f"configs/autoresearch/20260907/wave{args.wave:02d}.json").read_text())["lanes"]
inputs = {}
results = []
all_rows = {}
for precision, lanes in [("bfloat16", [0, 2]), ("float32", [1, 3])]:
    rows = []
    observed_precision = []
    for lane in lanes:
        status_path = root / f"lane{lane}-status.json"
        if json.loads(status_path.read_text())["status"] != "pass":
            raise ValueError("Incomplete precision diagnostic")
        config_path = root / f"lane{lane}/config.yaml"
        config = yaml.safe_load(config_path.read_text())
        if config["benchmark"]["precision"] != precision:
            raise ValueError("Runtime precision mismatch")
        spec = specs[lane]
        campaign_path = root / f"lane{lane}/mapper-campaign.json"
        campaign = json.loads(campaign_path.read_text())
        if "runtime_precision" in campaign:
            dtype = f"torch.{precision}"
            if campaign["runtime_precision"] != precision or any(
                campaign[key] != [dtype]
                for key in ["target_parameter_dtypes", "drafter_parameter_dtypes"]
            ) or any(v != [dtype] for v in campaign["mapper_parameter_dtypes"].values()):
                raise ValueError("Actual parameter dtypes differ from configuration")
            if precision == "float32" and campaign["matmul_allow_tf32"]:
                raise ValueError("FP32 diagnostic enabled TF32")
            observed_precision.append(True)
        else:
            observed_precision.append(False)
        manifest_path = Path(config["benchmark"]["manifest_path"])
        expected = {r["problem_id"] for r in json.loads(manifest_path.read_text())["records"]}
        path = root / f"lane{lane}/benchmark-rank0.jsonl"
        shard = [json.loads(x) for x in path.read_text().splitlines()]
        if {r["problem_id"] for r in shard} != expected or len(expected) != 8:
            raise ValueError("Precision shard differs from declaration")
        for r in shard:
            if r["method"] == "relay_base" and r["mapper_checkpoint_sha256"] != spec["checkpoint_sha256"][spec["checkpoint"]]:
                raise ValueError("Mapper checkpoint changed")
        rows.extend(shard)
        for p in [path, config_path, manifest_path, status_path, campaign_path]:
            inputs[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    summary = summarize(rows, reference="native_ar")
    if summary["requests"] != 16:
        raise ValueError("Missing paired requests")
    all_rows[precision] = {(r["problem_id"], r["method"]): r for r in rows}
    results.append(dict(precision=precision, target=config["target"],
                        observed_parameter_precision=all(observed_precision), **summary))
if all_rows["bfloat16"].keys() != all_rows["float32"].keys():
    raise ValueError("Precisions used different request/method pairs")
cross_precision = {
    method: sum(r["output_hash"] == all_rows["float32"][key]["output_hash"]
                for key, r in all_rows["bfloat16"].items() if key[1] == method)
    for method in ["native_ar", "relay_base"]
}
(Path("reports/autoresearch-20260907") / args.output).write_text(json.dumps(
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
