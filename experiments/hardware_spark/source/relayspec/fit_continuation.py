"""Validate exact continuation without resetting optimizer or sample position."""


def validate_continuation(state, trial, cache_hash):
    if state["feature_cache_index_sha256"] != cache_hash:
        raise ValueError("continuation uses a different feature cache")
    previous = state["completed_trial"]
    # Only execution/provenance fields and the longer endpoint may change.
    ignored = {
        "name",
        "study",
        "steps",
        "checkpoint_steps",
        "budget_panels",
        "cache_backend",
        "device_cache",
        "resume_from",
        "resume_sha256",
    }
    for key in (previous.keys() | trial.keys()) - ignored:
        if previous.get(key) != trial.get(key):
            raise ValueError(f"continuation changes scientific setting: {key}")
    start = state["steps"]
    if start != previous["steps"] or not 0 < start < trial["steps"]:
        raise ValueError("continuation must extend a completed trajectory")
    if any(step <= start for step in trial["checkpoint_steps"]):
        raise ValueError("continuation checkpoints must follow its parent endpoint")
    if not state["optimizer"]["state"] or state["tokens_seen"] <= 0:
        raise ValueError("continuation lacks optimizer or data position")
    return start
