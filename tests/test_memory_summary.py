from __future__ import annotations

import importlib.util
from pathlib import Path


def load_summarizer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_memory_pair.py"
    spec = importlib.util.spec_from_file_location("memory_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(method: str, problem_id: str, before: int, peak: int) -> dict:
    return {
        "method": method,
        "problem_id": problem_id,
        "repetition": 0,
        "output_tokens": 100,
        "request_seconds": 2.0,
        "allocated_memory_before_bytes": before,
        "reserved_memory_before_bytes": before + 10,
        "peak_allocated_memory_bytes": peak,
        "peak_reserved_memory_bytes": peak + 10,
    }


def test_summarize_memory_pair_reports_isolated_residency_delta() -> None:
    summarize = load_summarizer().summarize_memory_pair
    gib = 1 << 30
    source = [
        row("source_reuse_eagle3", "p0", 10 * gib, 11 * gib),
        row("source_reuse_eagle3", "p1", 10 * gib, 12 * gib),
    ]
    relay = [
        row("relay_eagle3", "p0", 6 * gib, 7 * gib),
        row("relay_eagle3", "p1", 6 * gib, 8 * gib),
    ]

    result = summarize(source, relay, expected_requests=2)

    assert result["paired_problem_ids"] == 2
    assert result["allocated_before_saved_bytes"] == 4 * gib
    assert result["allocated_before_saved_gib"] == 4.0
    assert result["peak_allocated_saved_bytes"] == 4 * gib
    assert result["source_tokens_per_second"] == 50.0
    assert result["relay_tokens_per_second"] == 50.0
