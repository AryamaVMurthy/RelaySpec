import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "subset", Path(__file__).parents[1] / "scripts/evaluate_evalplus_subset.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_subset_preserves_every_official_case_and_rejects_missing_or_duplicate_ids():
    original = {
        "a": {"base_input": [1], "plus_input": [2, 3]},
        "b": {"base_input": [5]},
    }
    assert module.select_tasks(original, ["a"]) == {"a": original["a"]}
    assert module.select_tasks(original, ["a"])["a"] is original["a"]
    for ids in [[], ["a", "a"], ["missing"]]:
        with pytest.raises(ValueError):
            module.select_tasks(original, ids)
