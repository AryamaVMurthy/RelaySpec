import copy
import json
from pathlib import Path

import pytest

from relayspec.task_complexity import input_strata


def protocol():
    return json.loads(
        Path("configs/submission/scaling/task-complexity-development.json").read_text()
    )


def record():
    return {"benchmark": "math500", "metadata": {"level": 3, "subject": "Algebra"}}


def test_assignment_ignores_generated_outcomes():
    original = record()
    changed = {
        **original,
        "correct": False,
        "output_tokens": 2048,
        "request_seconds": 1000,
    }
    assert input_strata(original, 128, protocol()) == input_strata(
        changed, 128, protocol()
    )
    assert input_strata(original, 129, protocol())["input_length"] == "input_129_256"


def test_bad_or_overlapping_labels_fail_closed():
    config = protocol()
    config["difficulty"]["level_1_2"].append(3)
    with pytest.raises(ValueError, match="disjoint"):
        input_strata(record(), 64, config)
    invalid = copy.deepcopy(record())
    invalid["metadata"]["level"] = True
    with pytest.raises(ValueError, match="difficulty"):
        input_strata(invalid, 64, protocol())
    with pytest.raises(ValueError, match="positive"):
        input_strata(record(), 0, protocol())


@pytest.mark.parametrize("tokens", [1, 64, 65, 128, 129, 256, 257, 4096])
def test_all_levels_and_length_boundaries_have_one_group(tokens):
    config = protocol()
    for level in range(1, 6):
        row = record()
        row["metadata"]["level"] = level
        assigned = input_strata(row, tokens, config)
        assert level in config["difficulty"][assigned["difficulty"]]
        assert assigned["subject"] == "Algebra"
