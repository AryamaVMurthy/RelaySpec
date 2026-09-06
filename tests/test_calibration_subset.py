import pytest

from relayspec.calibration_subset import training_window


def test_window_counts_training_records_and_excludes_interleaved_validation():
    entries = [
        {"split": "validation", "id": "v0"},
        {"split": "train", "id": "a"},
        {"split": "validation", "id": "v1"},
        {"split": "train", "id": "b"},
        {"split": "train", "id": "c"},
    ]
    assert [e["id"] for e in training_window(entries, 2, 1)] == ["b", "c"]
    assert [e["id"] for e in training_window(entries, 2)] == ["a", "b"]
    assert len(entries) == 5


@pytest.mark.parametrize("offset", [-1, 1.5, True, 2])
def test_invalid_window_fails_instead_of_silently_truncating(offset):
    with pytest.raises(ValueError):
        training_window([{"split": "train"}] * 3, 2, offset)
