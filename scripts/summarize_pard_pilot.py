"""Require all four released-PARD pilot workers before declaring readiness."""

import hashlib
import json
import math
import os
from pathlib import Path


def main():
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    config_sha = hashlib.sha256(
        Path("configs/submission/baselines/pard-pilot.yaml").read_bytes()
    ).hexdigest()
    gates, rows = [], []
    for rank in range(4):
        gate_path = output / f"pard-gate-rank{rank}.json"
        gate = json.loads(gate_path.read_text())
        path = output / f"pard-rank{rank}.json"
        if (
            gate["status"] != "pass"
            or gate["rank"] != rank
            or gate["config_sha256"] != config_sha
            or gate["rows_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("incomplete or inconsistent PARD worker gate")
        worker = json.loads(path.read_text())["rows"]
        if set(worker) != {"eager_ar", "pard", "pard_duplicate"}:
            raise ValueError("missing PARD pilot comparison arm")
        if len({tuple(row["token_ids"]) for row in worker.values()}) != 1:
            raise ValueError("PARD worker output equality is not supported by raw rows")
        if any(
            not math.isfinite(row["request_seconds"])
            or row["request_seconds"] <= 0
            or not 0 < row["output_tokens"] <= 128
            for row in worker.values()
        ):
            raise ValueError("invalid PARD pilot measurement")
        rows.extend(worker.values())
        gates.append(
            {
                "rank": rank,
                "gate_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
            }
        )
    if len({row["problem_id"] for row in rows}) != 4:
        raise ValueError("PARD pilot did not evaluate four distinct requests")
    methods = {}
    for method in ("eager_ar", "pard", "pard_duplicate"):
        selected = [row for row in rows if row["method"] == method]
        methods[method] = {
            "requests": len(selected),
            "aggregate_request_tokens_per_second": sum(
                row["output_tokens"] for row in selected
            )
            / sum(row["request_seconds"] for row in selected),
            "raw_extra_tokens": sum(
                row["raw_output_tokens"] - row["output_tokens"] for row in selected
            ),
            "max_peak_gpu_bytes": max(row["peak_gpu_bytes"] for row in selected),
        }
    (output / "pard-pilot-gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "workers": gates,
                "methods": methods,
                "config_sha256": config_sha,
                "scope": "Four-request128-token resource/correctness pilot only. Released PARD "
                "and eager AR use the same target, static-cache runtime and tokenized "
                "prompts. No full-answer quality conclusion or cross-engine speed ranking.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
