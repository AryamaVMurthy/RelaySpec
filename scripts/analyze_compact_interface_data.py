"""Audit within-role sensitivity to 16 versus 128 calibration records."""

import json
from pathlib import Path

from relayspec.ar_paper_evidence import summarize
from relayspec.cached_fit_evidence import audit_fit_artifacts, digest

root = Path("reports/autoresearch-20260907/run-28530")
wave_path = Path("configs/autoresearch/20260907/wave40.json")
wave = json.loads(wave_path.read_text())
cache_sha = "a2fd012df1da5554f8e211cf0b60aec098d3754d866bc50e9623f3117f43f8e5"
inputs = {str(wave_path): digest(wave_path)}
ledger = json.loads(Path("reports/autoresearch-20260907/jobs.json").read_text())
job = next(j for j in ledger["jobs"] if j["id"] == 28530)
if (root / "source-commit.txt").read_text().strip() != job["source_commit"]:
    raise ValueError("Unmatched source snapshot")
inputs[str(root / "source-commit.txt")] = digest(root / "source-commit.txt")
common_ids = None
results = []
for i, spec in enumerate(wave["lanes"]):
    lane = root / f"lane{i}"
    paths = [
        root / f"lane{i}-status.json",
        root / f"lane{i}-spec.json",
        lane / "transform.json",
        lane / "mapper-campaign.json",
        lane / "benchmark-rank0.jsonl",
    ]
    status, executed, transform, campaign = [
        json.loads(p.read_text()) for p in paths[:4]
    ]
    if status["status"] != "pass" or executed != spec:
        raise ValueError("Incomplete or undeclared data cell")
    trial = transform["fit_trial"]
    declared = {
        k: v
        for k, v in trial.items()
        if k not in ["native_teacher", "native_teacher_sha256"]
    }
    if (
        declared != spec["trial"]
        or bool(trial.get("native_teacher")) != spec["native_teacher"]
    ):
        raise ValueError("Executed fitting trial differs from declaration")
    fit = audit_fit_artifacts(
        lane / "fitting" / trial["name"], trial, cache_sha256=cache_sha
    )
    for cp, sha in spec["checkpoint_sha256"].items():
        found = [v for v in campaign["variants"].values() if v["checkpoint"] == cp]
        if len(found) != 1 or found[0]["sha256"] != sha:
            raise ValueError("512-record reference hash mismatch")
    rows = [json.loads(x) for x in paths[-1].read_text().splitlines()]
    ids = {r["problem_id"] for r in rows}
    if (
        len(rows) != 16
        or len(ids) != 8
        or {(r["problem_id"], r["method"]) for r in rows}
        != {(p, m) for p in ids for m in ["relay_base", "relay_reduced"]}
    ):
        raise ValueError("Incomplete paired screen")
    if common_ids is None:
        common_ids = ids
    elif ids != common_ids:
        raise ValueError("Interface roles used different questions")
    summary = summarize(rows, reference="relay_base")
    result = dict(
        role="native" if spec["native_teacher"] else "retargeted",
        records=trial["distinct_examples"],
        fit=fit,
        decoding=summary,
    )
    results.append(result)
    for p in paths:
        inputs[str(p)] = digest(p)
    for name, sha in fit["source_sha256"].items():
        inputs[str(lane / "fitting" / trial["name"] / name)] = sha
out = dict(
    input_sha256=inputs,
    results=results,
    scope="Fixed8192 updates, common record prefixes and layers25/33. Each role uses its original normalization and teacher, with different output widths. Decoding retention uses each role's own512-record two-layer checkpoint on eight exposed GSM8K questions. Feature validation uses128 records; training diagnostics use the same first16 records for both fitting-set sizes. No cross-role absolute error ranking, fresh quality confirmation, or feature-extraction cost measurement.",
)
Path("reports/autoresearch-20260907/compact-interface-data-summary.json").write_text(
    json.dumps(out, indent=2) + "\n"
)
for r in results:
    print(
        r["role"],
        r["records"],
        r["fit"]["train_objective"],
        r["fit"]["validation_objective"],
        r["decoding"]["methods"]["relay_reduced"]["throughput_ratio"],
    )
