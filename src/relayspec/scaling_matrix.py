"""Predeclared crossed fitting study, with shared continuous trajectories."""

import math


def trial(architecture, width, n, *, seed=1729, l2=0.0, decay=0.0, study="core"):
    """A checkpoint can serve two budget panels when the trajectory is identical."""
    steps = max(8192, n)
    epoch_steps = {
        int(n * epoch / 4) for epoch in [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
    }
    name = f"{architecture}{width or ''}-n{n}-s{seed}"
    if l2:
        name += f"-l2-{l2:g}"
    if decay:
        name += f"-wd-{decay:g}"
    result = {
        "name": name,
        "study": study,
        "architecture": architecture,
        "seed": seed,
        "steps": steps,
        "distinct_examples": n,
        "learning_rate": 6e-4,
        "l2_weight": l2,
        "weight_decay": decay,
        "checkpoint_steps": sorted(
            {128, 512, 2048, 8192, n} | {s for s in epoch_steps if s <= steps}
        ),
        "validation_records": 1024,
        "diagnostic_train_records": min(256, n),
        "feature_objective": "relative_interface_mse",
        "normalize_input": True,
        "budget_panels": {
            "fixed_exposure": {"checkpoint_step": 8192, "presentations": 32768},
            "fixed_passes": {"checkpoint_step": n, "presentations": 4 * n},
        },
    }
    if width is not None:
        result["width"] = width
    return result


def primary_matrix():
    architectures = [("dense", None)] + [
        (family, width)
        for family in ["factorized", "mlp"]
        for width in [128, 512, 2048]
    ]
    cells = [
        trial(family, width, n)
        for n in [128, 512, 2048, 8192, 32768]
        for family, width in architectures
    ]
    cells += [
        trial(family, width, n, study="extended_capacity")
        for n in [512, 32768]
        for family in ["factorized", "mlp"]
        for width in [64, 256, 1024, 4096]
    ]
    return cells


def regularization_matrix():
    return [
        trial(family, width, n, l2=l2, decay=decay, study="regularization")
        for n in [512, 32768]
        for family, width in [("dense", None), ("factorized", 512), ("mlp", 512)]
        for l2, decay in [(v, 0) for v in [1e-7, 1e-6, 1e-5, 1e-4]]
        + [(0, v) for v in [1e-4, 1e-3, 1e-2]]
    ]


def validate_trial(t):
    n, steps = t["distinct_examples"], t["steps"]
    if n < 4 or n % 4 or steps < 1:
        raise ValueError("fitting data and updates must respect batch four")
    if t["architecture"] not in {"dense", "factorized", "mlp"}:
        raise ValueError("unknown mapper architecture")
    if t["architecture"] != "dense" and (
        not isinstance(t.get("width"), int) or t["width"] < 1
    ):
        raise ValueError("factored and MLP widths must be positive integers")
    if any(not math.isfinite(t[k]) or t[k] < 0 for k in ["l2_weight", "weight_decay"]):
        raise ValueError("regularization must be finite and nonnegative")
    if t["l2_weight"] and t["weight_decay"]:
        raise ValueError("explicit L2 and AdamW decay are separate arms")
    if any(s < 1 or s > steps for s in t["checkpoint_steps"]):
        raise ValueError("checkpoint outside the trajectory")
    for panel in t.get("budget_panels", {}).values():
        if panel["checkpoint_step"] not in t["checkpoint_steps"]:
            raise ValueError("budget panel has no saved checkpoint")
        if panel["presentations"] != panel["checkpoint_step"] * 4:
            raise ValueError("budget panel miscounts presented records")


def four_gpu_batches(cells):
    """Group comparable fit costs; never duplicate a cell merely to fill a GPU."""
    if len(cells) % 4 or len({t["name"] for t in cells}) != len(cells):
        raise ValueError("batches require unique trials in multiples of four")
    for t in cells:
        validate_trial(t)

    def cost(t):
        parameters = (
            20480 * 2560 if t["architecture"] == "dense" else t["width"] * 23040
        )
        # I/O imposes a lower bound on the cost of even a very small mapper.
        return t["steps"] * (parameters + 12_000_000)

    ordered = sorted(cells, key=lambda t: (cost(t), t["name"]))
    return [ordered[i : i + 4] for i in range(0, len(ordered), 4)]
