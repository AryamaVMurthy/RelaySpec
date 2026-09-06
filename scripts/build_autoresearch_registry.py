"""Index every collected research lane, retaining failed and negative experiments."""

import hashlib
import json
from pathlib import Path

root = Path("reports/autoresearch-20260907")
ledger_path = root / "jobs.json"
ledger = json.loads(ledger_path.read_text())
records = []
pipelines = []
collector_path = root / "collector-status.json"
collector = json.loads(collector_path.read_text()) if collector_path.exists() else {}
inputs = {str(ledger_path): hashlib.sha256(ledger_path.read_bytes()).hexdigest()}
for job in ledger["jobs"]:
    run = root / f"run-{job['id']}"
    if job.get("kind", "").startswith("boundary_"):
        state = collector.get(str(job["id"]), {})
        if not state.get("collected"):
            continue
        gate_name = {
            "boundary_pipeline_pilot": "pilot-gate.json",
            "boundary_extraction": "extraction-complete.json",
            "boundary_capacity_pilot": "batch-gate.json",
            "boundary_full_fit": "batch-gate.json",
            "boundary_evaluation": "campaign-gate.json",
        }[job["kind"]]
        path = run / gate_name
        gate = json.loads(path.read_text()) if path.exists() else {}
        if path.exists():
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        pipelines.append(dict(job=job["id"], wave=job["wave"], kind=job["kind"],
                              status="pass" if state["state"] == "COMPLETED" and gate.get("status") == "pass" else "fail",
                              slurm_state=state["state"], elapsed=state["elapsed"],
                              gate=gate_name, details=gate))
        continue
    for lane in range(4):
        spec_path = run / f"lane{lane}-spec.json"
        status_path = run / f"lane{lane}-status.json"
        if not status_path.exists():
            continue
        spec = json.loads(spec_path.read_text())
        status = json.loads(status_path.read_text())
        item = dict(
            job=job["id"],
            wave=job["wave"],
            lane=lane,
            family=spec["family"],
            transform=spec["transform"],
            hypothesis=spec["hypothesis"],
            **{k: status[k] for k in ["status", "elapsed_seconds", "exit_code"]},
        )
        inputs[str(spec_path)] = hashlib.sha256(spec_path.read_bytes()).hexdigest()
        inputs[str(status_path)] = hashlib.sha256(status_path.read_bytes()).hexdigest()
        path = run / f"lane{lane}" / "research-result.json"
        if path.exists():
            result = json.loads(path.read_text())
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            item.update(
                scope=result["scope"],
                summary=result["summary"],
                duplicate_control=result.get("duplicate_control"),
            )
            if result.get("duplicate_control", {}).get("status") in {"fail", "failed"}:
                item["status"] = "control_failed"
        elif item["status"] == "pass":
            raise ValueError(f"Completed lane has no scientific summary: {run}/{lane}")
        records.append(item)
output = dict(
    ledger_sha256=inputs[str(ledger_path)],
    input_sha256=inputs,
    lanes=records,
    pipelines=pipelines,
    scope="All collected terminal lanes, including failures and negative results. Missing lanes are not assumed completed. Repeated baselines and overlapping development questions are not independent experiments. Per-lane confirmation summaries must not replace the aggregated fixed analyses.",
)
(root / "experiment-registry.json").write_text(json.dumps(output, indent=2) + "\n")
lines = [
    "# Autoresearch experiment registry",
    "",
    output["scope"],
    "",
    "| Wave / lane | Job | Family / transform | Status | Seconds | Requests | Candidate throughput retention |",
    "|---|---:|---|---|---:|---:|---|",
]
for r in records:
    summary = r.get("summary", {})
    methods = summary.get("methods", {})
    arms = "; ".join(
        f"{m}: {100 * v['throughput_ratio']:.1f}%"
        for m, v in methods.items()
        if m
        not in [
            "relay_base",
            "native_ar",
            "native_target_dflash",
            "native_target_eagle3",
            "relay_duplicate",
        ]
    )
    lines.append(
        f"| {r['wave']} / {r['lane']} | {r['job']} | {r['family']} / {r['transform']} | {r['status']} | {r['elapsed_seconds']:.1f} | {summary.get('requests', '—')} | {arms or '—'} |"
    )
lines += ["", "## Four-GPU pipeline stages", "",
          "These use stage-specific completion gates, not independent lane summaries.", "",
          "| Wave | Job | Stage | Status | Elapsed | Gate |",
          "|---:|---:|---|---|---|---|"]
for p in pipelines:
    lines.append(f"| {p['wave']} | {p['job']} | {p['kind']} | {p['status']} | {p['elapsed']} | {p['gate']} |")
(root / "EXPERIMENTS.md").write_text("\n".join(lines) + "\n")
print(
    json.dumps(
        dict(
            terminal_lanes=len(records),
            passed=sum(r["status"] == "pass" for r in records),
            other=sum(r["status"] != "pass" for r in records),
            terminal_pipelines=len(pipelines),
        )
    )
)
