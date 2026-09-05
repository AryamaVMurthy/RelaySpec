from __future__ import annotations

import importlib.util
from pathlib import Path


def load_exporter():
    path = Path(__file__).resolve().parents[1] / "scripts" / "export_evalplus.py"
    spec = importlib.util.spec_from_file_location("evalplus_exporter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_code_prefers_python_fence() -> None:
    extract_code = load_exporter().extract_code
    text = "Explanation\n```python\ndef add(a, b):\n    return a + b\n```\nDone"

    assert extract_code(text) == "def add(a, b):\n    return a + b"


def test_evalplus_task_id_normalizes_mbpp_prefix() -> None:
    evalplus_task_id = load_exporter().evalplus_task_id

    assert evalplus_task_id("mbpp", "mbpp/42") == "Mbpp/42"
    assert evalplus_task_id("humaneval", "HumanEval/1") == "HumanEval/1"
