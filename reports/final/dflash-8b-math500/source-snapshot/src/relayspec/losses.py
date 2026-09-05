from __future__ import annotations

import torch
import torch.nn.functional as F


def relative_interface_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Coefficient-free per-token error relative to frozen interface energy."""
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target interface tensors must have equal shapes"
        )
    if prediction.ndim < 2:
        raise ValueError("interface tensors must include token and feature dimensions")
    prediction_float = prediction.float()
    target_float = target.detach().float()
    squared_error = (prediction_float - target_float).square().sum(dim=-1)
    target_energy = target_float.square().sum(dim=-1)
    denominator = target_energy.clamp_min(torch.finfo(target_energy.dtype).tiny)
    return (squared_error / denominator).mean()


def interface_alignment_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    objective: str,
    historical_cosine_weight: float | None = None,
) -> torch.Tensor:
    """Select a registered interface objective without hidden coefficients."""
    if objective == "relative_interface_mse":
        return relative_interface_mse(prediction, target)
    target = target.detach()
    if objective == "raw_mse":
        return F.mse_loss(prediction.float(), target.float())
    if objective == "historical_mse_cosine":
        if historical_cosine_weight is None:
            raise ValueError(
                "historical_cosine_weight must be explicit for the historical loss"
            )
        mse = F.mse_loss(prediction.float(), target.float())
        cosine = 1 - F.cosine_similarity(
            prediction.float(), target.float(), dim=-1
        ).mean()
        return mse + float(historical_cosine_weight) * cosine
    raise ValueError(f"unknown interface objective: {objective}")


def proposal_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher proposal logits must have equal shapes")
    if temperature <= 0:
        raise ValueError("proposal KL temperature must be positive")
    teacher_log_probabilities = F.log_softmax(
        teacher_logits.detach().float() / temperature,
        dim=-1,
    )
    teacher_probabilities = teacher_log_probabilities.exp()
    student_log_probabilities = F.log_softmax(
        student_logits.float() / temperature,
        dim=-1,
    )
    token_kl = (
        teacher_probabilities * (teacher_log_probabilities - student_log_probabilities)
    ).sum(dim=-1)
    return token_kl.mean() * (temperature**2)
