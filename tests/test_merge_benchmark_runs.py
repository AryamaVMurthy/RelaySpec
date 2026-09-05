from __future__ import annotations

import importlib.util
from pathlib import Path


def load_merger():
    path = Path(__file__).resolve().parents[1] / "scripts" / "merge_benchmark_runs.py"
    spec = importlib.util.spec_from_file_location("benchmark_merger", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(problem_id: str, method: str) -> dict[str, object]:
    return {
        "benchmark": "mbpp",
        "problem_id": problem_id,
        "method": method,
        "repetition": 0,
    }


def test_merge_rows_forms_exact_full_cartesian_evaluation() -> None:
    merge_rows = load_merger().merge_rows
    methods = ("native_ar", "relay_f")
    first = [row("Mbpp/1", method) for method in methods]
    remainder = [row("Mbpp/2", method) for method in methods]

    merged = merge_rows(
        first,
        remainder,
        expected_problem_ids={"Mbpp/1", "Mbpp/2"},
        methods=methods,
    )

    assert len(merged) == 4
    assert {(r["problem_id"], r["method"]) for r in merged} == {
        (problem_id, method)
        for problem_id in ("Mbpp/1", "Mbpp/2")
        for method in methods
    }


def test_merge_rows_rejects_overlap() -> None:
    merge_rows = load_merger().merge_rows
    duplicate = row("Mbpp/1", "native_ar")

    try:
        merge_rows(
            [duplicate],
            [duplicate],
            expected_problem_ids={"Mbpp/1"},
            methods=("native_ar",),
        )
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("overlapping benchmark shards were accepted")


def test_merge_rows_rejects_missing_task_method_pair() -> None:
    merge_rows = load_merger().merge_rows

    try:
        merge_rows(
            [row("Mbpp/1", "native_ar")],
            [],
            expected_problem_ids={"Mbpp/1"},
            methods=("native_ar", "relay_f"),
        )
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("incomplete benchmark merge was accepted")
