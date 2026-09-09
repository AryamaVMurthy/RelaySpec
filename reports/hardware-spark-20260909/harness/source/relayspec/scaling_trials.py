"""Preflight contracts for short mapper trials and later full study cells."""

import math

TRIAL_BUDGETS = {
    "pilot": {"steps": 16, "max_prompts": 8, "max_new_tokens": 128},
    "directional": {"steps": 1024, "max_prompts": 16, "max_new_tokens": 256},
}


def validate_trial(spec):
    for field in (
        "name",
        "hypothesis",
        "falsifier",
        "stage",
        "architecture",
        "distinct_examples",
        "seed",
    ):
        if field not in spec:
            raise ValueError(f"missing trial field: {field}")
    if spec["stage"] not in TRIAL_BUDGETS:
        raise ValueError("trial stage must be pilot or directional")
    if spec["architecture"] not in {"dense", "factorized", "mlp"}:
        raise ValueError("unknown mapper architecture")
    n = spec["distinct_examples"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 4 or n % 4:
        raise ValueError("distinct budget must be a positive multiple of four")
    width = spec.get("width")
    if spec["architecture"] == "dense" and width is not None:
        raise ValueError("dense map has no width hyperparameter")
    if spec["architecture"] != "dense" and (
        not isinstance(width, int) or isinstance(width, bool) or width < 1
    ):
        raise ValueError("factorized/MLP map requires a positive width")
    l2 = float(spec.get("l2_weight", 0.0))
    decay = float(spec.get("weight_decay", 0.0))
    if any(not math.isfinite(v) or v < 0 for v in (l2, decay)) or (l2 and decay):
        raise ValueError("use finite nonnegative L2 or weight decay in separate arms")
    if spec.get("timeout_seconds", 540) != 540:
        raise ValueError("short trials have an immutable 540-second process budget")
    budget = TRIAL_BUDGETS[spec["stage"]]
    return {
        **budget,
        "available_records": n,
        "distinct_records_seen": min(n, budget["steps"] * 4),
        "record_presentations": budget["steps"] * 4,
        "role": "development; capped outputs do not establish full-answer quality",
        "timeout_seconds": 540,
    }
