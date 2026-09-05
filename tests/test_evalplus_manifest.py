from __future__ import annotations

import importlib.util
from pathlib import Path


def load_builder():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_evalplus_mbpp_manifest.py"
    )
    spec = importlib.util.spec_from_file_location("evalplus_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evalplus_mbpp_records_are_scoreable_and_include_base_assertions() -> None:
    problems = {
        "Mbpp/1": {
            "prompt": '"""Add two integers.\nassert add(1, 2) == 3\n"""',
            "entry_point": "add",
            "assertion": "assert add(1, 2) == 3\nassert add(-1, 1) == 0",
        },
        "Mbpp/2": {
            "prompt": '"""Multiply two integers.\nassert mul(2, 3) == 6\n"""',
            "entry_point": "mul",
            "assertion": "assert mul(2, 3) == 6",
        },
    }

    records = load_builder().mbpp_evalplus_records(problems, limit=2, seed=42)

    assert {record["problem_id"] for record in records} == {"Mbpp/1", "Mbpp/2"}
    assert all("Your code must satisfy all base tests:" in r["prompt"] for r in records)
    assert all(r["answer"]["entry_point"] in r["prompt"] for r in records)
    assert sum(len(r["answer"]["tests"]) for r in records) == 3


def test_evalplus_mbpp_records_rejects_too_large_subset() -> None:
    try:
        load_builder().mbpp_evalplus_records({}, limit=1, seed=42)
    except ValueError as error:
        assert "only 0" in str(error)
    else:
        raise AssertionError("builder accepted more tasks than available")


def test_exclude_task_ids_returns_exact_disjoint_remainder() -> None:
    exclude_task_ids = load_builder().exclude_task_ids
    records = [
        {"problem_id": "Mbpp/1"},
        {"problem_id": "Mbpp/2"},
        {"problem_id": "Mbpp/3"},
    ]

    remainder = exclude_task_ids(records, {"Mbpp/1", "Mbpp/3"})

    assert remainder == [{"problem_id": "Mbpp/2"}]
