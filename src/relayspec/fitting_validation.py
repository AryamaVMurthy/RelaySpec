"""Fixed, record-weighted fitting diagnostics; never used as task quality scores."""

import hashlib
import json

import torch
import torch.nn.functional as F

from relayspec.losses import interface_alignment_loss, relative_interface_mse


def text_identity(row):
    return hashlib.sha256(" ".join(row["problem"].split()).encode()).hexdigest()


def validate_fit_split(training_rows, validation_rows):
    if not validation_rows:
        raise ValueError("validation split must be nonempty")
    train_ids = {text_identity(r) for r in training_rows}
    val_ids = [text_identity(r) for r in validation_rows]
    if len(set(val_ids)) != len(val_ids) or train_ids.intersection(val_ids):
        raise ValueError("fitting and validation problems must be unique and disjoint")
    return {
        "training_records": len(training_rows),
        "validation_records": len(validation_rows),
        "validation_content_sha256": hashlib.sha256(
            json.dumps(validation_rows, sort_keys=True).encode()
        ).hexdigest(),
        "scope": "Fitting validation; not untouched decoding evaluation. Exact normalized problem overlap checked; near-overlap audit is separate.",
    }


@torch.no_grad()
def interface_diagnostics(
    relay,
    examples,
    *,
    device,
    objective,
    historical_cosine_weight=None,
    output_transform=None,
):
    """Cache entries are unpadded individual records, matching training weighting."""
    if not examples:
        raise ValueError("diagnostic examples must be nonempty")
    sums = dict(
        objective=0.0, relative_mse=0.0, cosine_error=0.0, relative_norm_error=0.0
    )
    tokens = 0
    for x, y in examples:
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            prediction = relay(x.to(device))
            if output_transform is not None:
                prediction = output_transform(prediction)
        y = y.to(device).float()
        prediction = prediction.float()
        values = {
            "objective": interface_alignment_loss(
                prediction,
                y,
                objective=objective,
                historical_cosine_weight=historical_cosine_weight,
            ),
            "relative_mse": relative_interface_mse(prediction, y),
            "cosine_error": (1 - F.cosine_similarity(prediction, y, dim=-1)).mean(),
            "relative_norm_error": (
                (prediction.norm(dim=-1) - y.norm(dim=-1)).abs()
                / y.norm(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
            ).mean(),
        }
        if not all(torch.isfinite(v) for v in values.values()):
            raise ValueError("nonfinite fitting validation metric")
        for key, value in values.items():
            sums[key] += value.item()
        tokens += y.shape[-2]
    return {
        "records": len(examples),
        "tokens": tokens,
        **{k: v / len(examples) for k, v in sums.items()},
    }
