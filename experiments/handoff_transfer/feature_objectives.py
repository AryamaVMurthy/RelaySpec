"""Unweighted distribution losses over fused feature coordinates, T=1.

Raw signed feature vectors are logits, not probability vectors. Standard
log_softmax defines their distributions; there is no position-dependent weight.
Forward KL is KL(source || mapped); reverse is KL(mapped || source).
"""
import torch
from torch.nn import functional as F


def feature_distribution_loss(prediction, target, objective):
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("Expected paired feature vectors of identical shape")
    log_q = F.log_softmax(prediction.float(), dim=-1)
    log_p = F.log_softmax(target.detach().float(), dim=-1)
    if objective == "feature_ce":
        return -(log_p.exp() * log_q).sum(-1)
    if objective == "forward_kl":
        return (log_p.exp() * (log_p - log_q)).sum(-1)
    if objective == "reverse_kl":
        return (log_q.exp() * (log_q - log_p)).sum(-1)
    raise ValueError(f"Unknown feature objective: {objective}")
