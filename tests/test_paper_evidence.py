from __future__ import annotations

import json
from pathlib import Path

import pytest

from relayspec.paper_evidence import verify_recorded_pairs


def _pair(tmp_path: Path):
    rows = [
        {
            "problem_id": "HumanEval/1",
            "repetition": 0,
            "benchmark": "other",
            "method": method,
            "request_seconds": seconds,
            "output_tokens": 10,
            "output_hash": "same",
        }
        for method, seconds in [("optimized_source_reuse", 1.0), ("relay_p", 2.0)]
    ]
    path = tmp_path / "benchmark-rank0.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    row = {
        "family_key": "dflash",
        "requests": 1,
        "source_tps": 10.0,
        "relay_tps": 5.0,
        "speedup": 0.5,
        "exact_rate": 1.0,
    }
    return rows, path, row


def test_valid_slowdown_is_retained(tmp_path: Path) -> None:
    _, _, row = _pair(tmp_path)
    result = verify_recorded_pairs(tmp_path, row, tmp_path, "other")
    assert result["relay_tps"] < result["source_tps"]
    assert result["paired_requests"] == 1


def test_missing_or_duplicate_arm_is_rejected(tmp_path: Path) -> None:
    rows, path, row = _pair(tmp_path)
    path.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="Incomplete measurement"):
        verify_recorded_pairs(tmp_path, row, tmp_path, "other")
    path.write_text("".join(json.dumps(r) + "\n" for r in [*rows, rows[0]]))
    with pytest.raises(ValueError, match="Duplicate raw row"):
        verify_recorded_pairs(tmp_path, row, tmp_path, "other")


def test_stale_figure_value_is_rejected(tmp_path: Path) -> None:
    _, _, row = _pair(tmp_path)
    row["relay_tps"] = 50.0
    with pytest.raises(ValueError, match="Stale throughput"):
        verify_recorded_pairs(tmp_path, row, tmp_path, "other")
