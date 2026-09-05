from __future__ import annotations

import importlib.util
from pathlib import Path


def load_builder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_eval_manifests.py"
    spec = importlib.util.spec_from_file_location("manifest_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stable_subset_is_reproducible_and_does_not_mutate_input() -> None:
    stable_subset = load_builder().stable_subset
    rows = [{"id": index} for index in range(20)]

    first = stable_subset(rows, limit=5, seed=42)
    second = stable_subset(rows, limit=5, seed=42)

    assert first == second
    assert len(first) == 5
    assert rows == [{"id": index} for index in range(20)]


def test_gsm8k_answer_uses_official_delimiter() -> None:
    gsm8k_answer = load_builder().gsm8k_answer

    assert gsm8k_answer("reasoning\n#### 1,234") == "1234"


def test_mbpp_prompt_includes_all_base_tests_and_function_name() -> None:
    mbpp_prompt = load_builder().mbpp_prompt

    prompt = mbpp_prompt(
        "Write a function to add two integers.",
        ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
    )

    assert prompt.startswith("Write a function to add two integers.")
    assert "Your code must satisfy these tests:" in prompt
    assert "assert add(1, 2) == 3" in prompt
    assert "assert add(-1, 1) == 0" in prompt


def test_mbpp_prompt_rejects_missing_tests() -> None:
    mbpp_prompt = load_builder().mbpp_prompt

    try:
        mbpp_prompt("Write a function.", [])
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("MBPP prompt accepted an empty test list")
