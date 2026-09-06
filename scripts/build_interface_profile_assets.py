"""Attribute completed fixed-comparison runtime to recorded CUDA/profile regions."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

root = Path("reports/autoresearch-20260907/run-28414")
totals = defaultdict(lambda: defaultdict(float))
inputs = {}
for path in sorted(root.glob("lane*/benchmark-rank0.jsonl")):
    inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for line in path.read_text().splitlines():
        r = json.loads(line)
        t = totals[r["method"]]
        t["requests"] += 1
        t["request_ms"] += r["request_seconds"] * 1000
        t["target_calls"] += r["target_calls"]
        for k, v in r["profile_regions_ms"].items():
            t[k] += v
if any(t["requests"] != 64 for t in totals.values()) or len(totals) != 2:
    raise ValueError("Incomplete profile comparison")
result = {}
for method, t in totals.items():
    result[method] = dict(t)
    result[method]["mapping_fraction"] = (t["relay"] + t["prefill_relay"]) / t[
        "request_ms"
    ]
    result[method]["verification_fraction"] = (
        t["verification_full_target"] / t["request_ms"]
    )
Path("reports/autoresearch-20260907/interface-profile-summary.json").write_text(
    json.dumps(
        dict(
            input_sha256=inputs,
            methods=result,
            scope="Recorded region attribution on the same fixed64-request comparison. Mapper includes prefill_relay plus relay. Target verification excludes prefill. Not an isolated kernel benchmark or memory measurement.",
        ),
        indent=2,
    )
    + "\n"
)
print(
    {
        k: {
            q: round(v[q], 6)
            for q in ["mapping_fraction", "verification_fraction", "target_calls"]
        }
        for k, v in result.items()
    }
)
