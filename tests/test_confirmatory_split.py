from __future__ import annotations

import importlib.util
from pathlib import Path


def load_builder():
    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "build_confirmatory_split.py"
    )
    spec = importlib.util.spec_from_file_location("confirmatory_split", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(problem_id: str, prompt: str) -> dict[str, str]:
    return {"benchmark": "math500", "problem_id": problem_id, "prompt": prompt}


def test_confirmatory_split_excludes_registered_prompt_by_normalized_hash() -> None:
    build_confirmatory_split = load_builder().build_confirmatory_split
    development = {"records": [record("old", "  same   prompt\ntext ")]}
    full = {
        "records": [
            record("math500/0", "same prompt text"),
            record("math500/1", "different"),
        ]
    }

    result = build_confirmatory_split(
        development,
        full,
        seed=1729,
        development_count=1,
    )

    assert result["excluded_problem_ids"] == ["math500/0"]
    assert [row["problem_id"] for row in result["records"]] == ["math500/1"]
