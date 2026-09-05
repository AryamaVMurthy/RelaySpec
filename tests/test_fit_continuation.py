import copy

import pytest

from relayspec.fit_continuation import validate_continuation


def test_resume_rejects_changed_data_optimizer_budget_and_cache():
    previous = dict(
        name="parent",
        steps=512,
        checkpoint_steps=[512],
        architecture="mlp",
        width=512,
        distinct_examples=512,
        seed=1729,
        learning_rate=6e-4,
        l2_weight=1e-6,
    )
    state = dict(
        completed_trial=previous,
        steps=512,
        tokens_seen=1000,
        feature_cache_index_sha256="cache",
        optimizer={"state": {0: {"step": 512}}},
    )
    trial = {**previous, "name": "continued", "steps": 1024, "checkpoint_steps": [1024]}
    assert validate_continuation(state, trial, "cache") == 512
    for key, value in [
        ("distinct_examples", 2048),
        ("seed", 1730),
        ("learning_rate", 1e-4),
        ("l2_weight", 0.0),
    ]:
        with pytest.raises(ValueError, match="scientific setting"):
            validate_continuation(state, {**trial, key: value}, "cache")
    with pytest.raises(ValueError, match="different feature cache"):
        validate_continuation(state, trial, "other")
    with pytest.raises(ValueError, match="extend"):
        validate_continuation(state, {**trial, "steps": 512}, "cache")
    with pytest.raises(ValueError, match="follow"):
        validate_continuation(
            state, {**trial, "checkpoint_steps": [512, 1024]}, "cache"
        )
    broken = copy.deepcopy(state)
    broken["optimizer"]["state"] = {}
    with pytest.raises(ValueError, match="lacks optimizer"):
        validate_continuation(broken, trial, "cache")
