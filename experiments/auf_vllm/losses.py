"""Exact current-prediction AUF and matched uniform token CE."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class TokenLoss:
    loss: torch.Tensor
    active: torch.Tensor
    correct: torch.Tensor


def token_loss(logits, labels, valid, objective="auf"):
    """Return ONE microbatch's active-token mean, not a mean of block means.

    The caller divides each microbatch loss by its accumulation count. Labels
    outside ``valid`` may contain the usual -100 sentinel and never enter CE.
    Greedy tests are detached and recomputed on every call.
    """
    if objective not in {"auf", "ce"}:
        raise ValueError(f"Unsupported token objective: {objective}")
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or valid.shape != labels.shape:
        raise ValueError("Expected logits[B,J,V] and labels/valid[B,J]")
    if valid.dtype != torch.bool or labels.dtype != torch.long:
        raise ValueError("valid must be bool and labels must be int64")
    if not valid.any():
        raise ValueError("Microbatch has no valid supervised positions")
    if ((labels[valid] < 0) | (labels[valid] >= logits.shape[-1])).any():
        raise ValueError("A valid label is outside the drafter vocabulary")
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite draft logits")
    with torch.no_grad():
        correct = logits.argmax(-1).eq(labels)
        if objective == "auf":
            inclusive = ((~valid) | correct).to(torch.int64).cumprod(-1)
            exclusive = torch.cat([torch.ones_like(inclusive[:, :1]), inclusive[:, :-1]], -1)
            active = valid & exclusive.bool()
        else:
            active = valid.clone()
    loss = F.cross_entropy(logits[active].float(), labels[active], reduction="sum") / active.sum()
    return TokenLoss(loss=loss, active=active, correct=correct)
