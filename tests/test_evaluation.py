from __future__ import annotations

from relayspec.evaluation import extract_math_answer, math_equal


def test_extract_math_answer_uses_last_nested_box() -> None:
    completion = r"work \boxed{1} more work \boxed{\frac{5}{2}}"

    assert extract_math_answer(completion) == r"\frac{5}{2}"


def test_extract_math_answer_falls_back_to_final_answer_phrase() -> None:
    completion = "After simplifying, the final answer is 49."

    assert extract_math_answer(completion) == "49"


def test_math_equal_handles_latex_fraction_and_decimal() -> None:
    assert math_equal(r"\frac{5}{4}", "1.25")


def test_math_equal_handles_unbraced_latex_fraction_numerator() -> None:
    assert math_equal(r"\frac{9}{19}", r"\frac9{19}")


def test_math_equal_handles_escaped_currency_marker() -> None:
    assert math_equal("36", r"\$36")


def test_math_equal_ignores_standard_measurement_units() -> None:
    assert math_equal("15", r"15\mbox{ cm}^2")


def test_math_equal_handles_equivalent_symbolic_expressions() -> None:
    assert math_equal("x^2-1", "(x-1)(x+1)")


def test_math_equal_rejects_different_values() -> None:
    assert not math_equal("18", "16")
