"""Check declared paired rows and verifier replicas before summarizing new runs."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import yaml

from relayspec.ar_paper_evidence import summarize
from relayspec.pard_adapter import trim_pard_tokens, verify_pard_decisions


def audit(run):
    pard = (run / "config.json").exists()
    config_path = run / ("config.json" if pard else "config.yaml")
    config = yaml.safe_load(config_path.read_text())
    bench = config if pard else config["benchmark"]
    manifest_path = Path(bench["manifest_path"])
    records = json.loads(manifest_path.read_text())["records"]
    records = [
        r
        for r in records
        if r["benchmark"] in (["math500"] if pard else bench["benchmarks"])
    ]
    random.Random(config["seed"]).shuffle(records)
    records = records[: bench["requests"] if pard else bench["max_prompts"]]
    by_id = {r["problem_id"]: r for r in records}
    expected = {(m, pid) for m in bench["methods"] for pid in by_id}
    files = sorted(run.glob("benchmark-rank*.jsonl"))
    if len(files) != 4:
        raise ValueError("four worker outputs required")
    rows = [json.loads(line) for p in files for line in p.read_text().splitlines()]
    if (
        len(rows) != len(expected)
        or {(r["method"], r["problem_id"]) for r in rows} != expected
    ):
        raise ValueError("paired method/request matrix incomplete")
    if {r["rank"] for r in rows} != {0, 1, 2, 3}:
        raise ValueError("four ranks required")
    for row in rows:
        if (
            not math.isfinite(row["request_seconds"])
            or row["request_seconds"] <= 0
            or row["output_tokens"] <= 0
        ):
            raise ValueError("invalid duration or token count")
        if pard:
            if row["method"] != "native_ar":
                proof = row["verification_replica"]
                verify_pard_decisions(
                    proof["token_ids"],
                    proof["accepted_lengths"],
                    proof["target_trace"],
                    draft_k=config["draft_k"],
                )
                if (
                    row["raw_output_ids"] != proof["token_ids"]
                    or row["acceptance_lengths"] != proof["accepted_lengths"]
                ):
                    raise ValueError("observed and timed outputs differ")
                if row["output_ids"] != trim_pard_tokens(
                    row["raw_output_ids"],
                    max_new_tokens=config["max_new_tokens"],
                    eos_token_id=151645,
                ):
                    raise ValueError("EOS/cap trimming mismatch")
        elif row["input_tokens"] != by_id[row["problem_id"]]["context_tokens"]:
            raise ValueError("context length differs from declaration")
    if pard:
        for rank in range(4):
            gate = json.loads((run / f"campaign-rank{rank}.json").read_text())
            if (
                gate["status"] != "pass"
                or gate["config_sha256"]
                != hashlib.sha256(config_path.read_bytes()).hexdigest()
            ):
                raise ValueError("worker gate mismatch")
    else:
        if json.loads((run / "completion-gate.json").read_text())["status"] != "pass":
            raise ValueError("completion gate missing")
    reference = "native_ar"
    summary = summarize(rows, reference=reference)
    memory = {}
    if not pard:
        for method in bench["methods"]:
            rs = [r for r in rows if r["method"] == method]
            memory[method] = {
                "mean_prefill_seconds": sum(
                    r["time_to_first_token_seconds"] for r in rs
                )
                / len(rs),
                "decode_tokens_per_second": sum(r["output_tokens"] for r in rs)
                / sum(r["decode_seconds"] for r in rs),
                "max_peak_allocated_gib": max(
                    r["peak_allocated_memory_bytes"] for r in rs
                )
                / 2**30,
                "max_incremental_peak_gib": max(
                    r["peak_allocated_memory_bytes"]
                    - r["allocated_memory_before_bytes"]
                    for r in rs
                )
                / 2**30,
            }
    result = {
        "status": "pass",
        "records": len(rows),
        "summary": summary,
        "sequence_metrics": memory,
        "input_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [config_path, manifest_path, *files]
        },
        "scope": "Paired development measurements. PARD uses verifier-observed replicas. Context memory includes co-resident controls; incremental peak is also recorded. No task-quality claim from throughput.",
    }
    (run / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    if pard:
        (run / "completion-gate.json").write_text(
            json.dumps({"status": "pass", "records": len(rows)}) + "\n"
        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    a = p.parse_args()
    result = audit(a.run)
    print(json.dumps({"status": result["status"], "records": result["records"]}))
