from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from tokenize import TokenError
from typing import Any


_FINAL_ANSWER = re.compile(
    r"(?:final\s+answer\s*(?:is|:)|answer\s*(?:is|:))\s*([^\n]+)",
    flags=re.IGNORECASE,
)


def _last_boxed(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    cursor = start + len(marker)
    depth = 1
    while cursor < len(text):
        character = text[cursor]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start + len(marker) : cursor]
        cursor += 1
    return None


def extract_math_answer(completion: str) -> str | None:
    boxed = _last_boxed(completion)
    if boxed is not None:
        return boxed.strip()
    matches = list(_FINAL_ANSWER.finditer(completion))
    if not matches:
        return None
    answer = matches[-1].group(1).strip()
    return answer.rstrip(". ,;:") or None


def _replace_latex_fraction(value: str) -> str:
    marker = r"\frac{"
    while marker in value:
        start = value.rfind(marker)
        numerator_start = start + len(marker)
        numerator_end = _matching_brace(value, numerator_start - 1)
        if numerator_end is None or numerator_end + 1 >= len(value):
            break
        if value[numerator_end + 1] != "{":
            break
        denominator_end = _matching_brace(value, numerator_end + 1)
        if denominator_end is None:
            break
        numerator = value[numerator_start:numerator_end]
        denominator = value[numerator_end + 2 : denominator_end]
        replacement = f"(({numerator})/({denominator}))"
        value = value[:start] + replacement + value[denominator_end + 1 :]
    return value


def _matching_brace(value: str, opening: int) -> int | None:
    depth = 0
    for index in range(opening, len(value)):
        if value[index] == "{":
            depth += 1
        elif value[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def normalize_math_answer(value: Any) -> str:
    normalized = str(value).strip()
    normalized = normalized.replace(r"\$", "")
    normalized = normalized.replace("$", "")
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    normalized = normalized.replace(r"\!", "").replace(r"\,", "")
    normalized = normalized.replace(r"\dfrac", r"\frac").replace(
        r"\tfrac", r"\frac"
    )
    normalized = normalized.replace("−", "-").replace("–", "-")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(
        r"\\frac([+-]?(?:\d+(?:\.\d+)?|[A-Za-z]))\{",
        r"\\frac{\1}{",
        normalized,
    )
    normalized = re.sub(
        r"\\(?:text|mbox)\{(?:cm|mm|m|km|in|ft|yd|units?)\}"
        r"(?:\^\{?\d+\}?)?",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = normalized.rstrip(". ,;:")
    if normalized.startswith("{") and normalized.endswith("}"):
        normalized = normalized[1:-1]
    return normalized


def _numeric_value(value: str) -> Decimal | None:
    candidate = _replace_latex_fraction(value)
    simple_fraction = re.fullmatch(r"\(\(([-+]?\d+(?:\.\d+)?)\)/\(([-+]?\d+(?:\.\d+)?)\)\)", candidate)
    if simple_fraction:
        denominator = Decimal(simple_fraction.group(2))
        if denominator == 0:
            return None
        return Decimal(simple_fraction.group(1)) / denominator
    try:
        return Decimal(candidate)
    except InvalidOperation:
        return None


def _symbolic_equal(left: str, right: str) -> bool:
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        def parse(value: str):
            candidate = _replace_latex_fraction(value)
            candidate = candidate.replace("^", "**")
            candidate = candidate.replace("{", "(").replace("}", ")")
            candidate = candidate.replace(r"\cdot", "*").replace(r"\times", "*")
            return parse_expr(
                candidate,
                transformations=(
                    *standard_transformations,
                    implicit_multiplication_application,
                ),
            )

        return bool(sympy.simplify(parse(left) - parse(right)) == 0)
    except (ImportError, TokenError, TypeError, ValueError, SyntaxError):
        return False


def math_equal(prediction: Any, reference: Any) -> bool:
    if prediction is None or reference is None:
        return False
    left = normalize_math_answer(prediction)
    right = normalize_math_answer(reference)
    if not left or not right:
        return False
    if left == right:
        return True
    left_number = _numeric_value(left)
    right_number = _numeric_value(right)
    if left_number is not None and right_number is not None:
        return abs(left_number - right_number) <= Decimal("1e-9")
    return _symbolic_equal(left, right)


def score_math_completion(completion: str, reference: Any) -> dict[str, Any]:
    prediction = extract_math_answer(completion)
    return {
        "predicted_answer": prediction,
        "correct": math_equal(prediction, reference),
    }
