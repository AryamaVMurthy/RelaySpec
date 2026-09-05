from __future__ import annotations


def test_run_python_tests_accepts_correct_solution() -> None:
    from relayspec.code_eval import run_python_tests

    result = run_python_tests(
        "def add(a, b):\n    return a + b",
        ["assert add(2, 3) == 5"],
        timeout_seconds=2,
    )

    assert result["passed"] is True


def test_run_python_tests_terminates_infinite_loop() -> None:
    from relayspec.code_eval import run_python_tests

    result = run_python_tests(
        "while True:\n    pass",
        ["assert True"],
        timeout_seconds=1,
    )

    assert result["passed"] is False
    assert result["status"] in {"timeout", "signal"}
