"""Recheck historical reserve against all currently collected local decoding records."""

import hashlib
import json
from pathlib import Path

root = Path("configs/submission/confirmation-gsm8k-20260906")
parent = root / "confirmation-data-gate.json"
gate = json.loads(parent.read_text())
reserve = root / "gsm8k-reserve-256.json"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


if gate["status"] != "pass" or sha(reserve) != gate["output_sha256"][reserve.name]:
    raise ValueError("Historical reserve provenance does not pass")
ids = {r["problem_id"] for r in json.loads(reserve.read_text())["records"]}
inputs, exposed = {}, {}
for repository in [Path("/home/aryamavmurthy/work/RelaySpec"), Path.cwd()]:
    for p in sorted((repository / "reports").rglob("benchmark-rank*.jsonl")):
        inputs[str(p)] = sha(p)
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get("problem_id") in ids:
                exposed.setdefault(r["problem_id"], []).append(str(p))
result = dict(
    status="pass" if not exposed else "exposed",
    reserve_sha256=sha(reserve),
    parent_gate_sha256=sha(parent),
    reserve_records=len(ids),
    exposed=exposed,
    input_sha256=inputs,
    scope="Exposure audit of collected local rank logs in both worktrees. Inherits exact/lexical fitting-data exclusion from the unchanged historical gate. Does not launch evaluation or authorize adaptive use of the reserve. Recheck at confirmation launch; any uncollected jobs require separate accounting.",
)
out = Path("reports/autoresearch-20260907/reserve-exposure-audit.json")
out.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {k: v for k, v in result.items() if k not in ["input_sha256", "exposed"]}
    )
)
print("scanned files:", len(inputs), "exposed reserve records:", len(exposed))
