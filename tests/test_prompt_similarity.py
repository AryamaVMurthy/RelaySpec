from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_auditor():
    path = ROOT / "scripts" / "audit_prompt_similarity.py"
    spec = importlib.util.spec_from_file_location("prompt_similarity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tokenizer_keeps_words_numbers_and_punctuation() -> None:
    tokenize = _load_auditor().unique_tokens

    assert tokenize("A + A = 2") == {"a", "+", "=", "2"}
    assert tokenize("x_1 and π") == {"x_1", "and", "π"}


def test_similarity_audit_joins_multiturn_records() -> None:
    audit = _load_auditor().audit_prompt_similarity
    fit = {
        "records": [
            {"problem": "add 2 + 3 now"},
            {"problem": "travel to hawaii"},
        ]
    }
    evaluation = {
        "records": [
            {
                "benchmark": "math500",
                "problem_id": "math500/0",
                "prompt": "add 2 + 3",
            },
            {
                "benchmark": "mtbench",
                "problem_id": "mtbench/0",
                "turns": ["travel", "to hawaii"],
            },
        ]
    }

    result = audit(fit, evaluation, thresholds=(0.8, 0.9, 0.95), top_k=1)

    assert result["fit_records"] == 2
    assert result["evaluation_records"] == 2
    assert result["by_benchmark"]["math500"]["maximum"] == 4 / 5
    assert result["by_benchmark"]["math500"]["counts"] == {
        "0.80": 1,
        "0.90": 0,
        "0.95": 0,
    }
    assert result["by_benchmark"]["mtbench"]["maximum"] == 1.0
    assert result["by_benchmark"]["mtbench"]["top_pairs"][0]["fit_record_index"] == 1


def test_repository_similarity_counts_match_recorded_manifests() -> None:
    module = _load_auditor()
    fit = module.read_json(ROOT / "configs" / "train_math_4096.json")
    evaluation = module.read_json(ROOT / "configs" / "eval_manifest_full_v4.json")

    result = module.audit_prompt_similarity(fit, evaluation)

    expected = {
        "math500": (0.9795918367346939, 47, 13, 4),
        "gsm8k": (0.3783783783783784, 0, 0, 0),
        "humaneval": (0.2903225806451613, 0, 0, 0),
        "mbpp": (1 / 3, 0, 0, 0),
        "mtbench": (4 / 7, 0, 0, 0),
    }
    for benchmark, values in expected.items():
        row = result["by_benchmark"][benchmark]
        maximum, count_80, count_90, count_95 = values
        assert abs(row["maximum"] - maximum) < 1e-12
        assert row["counts"] == {
            "0.80": count_80,
            "0.90": count_90,
            "0.95": count_95,
        }
