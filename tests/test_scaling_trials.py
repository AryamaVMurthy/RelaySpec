import pytest

from relayspec.scaling_trials import validate_trial


def spec(**kwargs):
    return dict(
        name="test",
        hypothesis="smaller map may retain acceptance",
        falsifier="acceptance collapses",
        stage="directional",
        architecture="factorized",
        width=512,
        distinct_examples=512,
        seed=1729,
        **kwargs,
    )


def test_trial_distinguishes_available_from_seen_records():
    s = spec()
    s.update(stage="pilot", distinct_examples=2048)
    b = validate_trial(s)
    assert b["available_records"] == 2048
    assert b["distinct_records_seen"] == 64
    assert b["timeout_seconds"] == 540


@pytest.mark.parametrize(
    "changes",
    [
        {"l2_weight": 1e-4, "weight_decay": 1e-3},
        {"l2_weight": float("nan")},
        {"width": -2},
        {"distinct_examples": 0},
        {"timeout_seconds": 900},
        {"architecture": "dense"},
        {"stage": "final"},
    ],
)
def test_invalid_trial_fails_before_gpu_launch(changes):
    s = spec()
    s.update(changes)
    with pytest.raises(ValueError):
        validate_trial(s)
