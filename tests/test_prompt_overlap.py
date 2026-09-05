from __future__ import annotations

import importlib.util
from pathlib import Path


def load_checker():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_prompt_overlap.py"
    spec = importlib.util.spec_from_file_location("prompt_overlap", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalization_preserves_mathematical_operators() -> None:
    normalize = load_checker().normalize_prompt

    assert normalize("  z^4 - z^2 + 1  ") != normalize("z^4 + z^2 + 1")
    assert normalize("same\n  prompt") == normalize("same prompt")


def test_overlap_audit_handles_single_and_multiturn_prompts() -> None:
    audit = load_checker().audit_prompt_overlap
    fit = {
        "records": [
            {"problem": "same math"},
            {"problem": "travel followup"},
        ]
    }
    evaluation = {
        "records": [
            {
                "benchmark": "math500",
                "problem_id": "math500/0",
                "prompt": "same   math",
            },
            {
                "benchmark": "mtbench",
                "problem_id": "mtbench/0",
                "turns": ["first turn", "travel followup"],
            },
            {
                "benchmark": "math500",
                "problem_id": "math500/1",
                "prompt": "different",
            },
        ]
    }

    result = audit(fit, evaluation)

    assert result["fit_records"] == 2
    assert result["evaluation_records"] == 3
    assert result["overlap_count"] == 2
    assert [row["problem_id"] for row in result["overlaps"]] == [
        "math500/0",
        "mtbench/0",
    ]
