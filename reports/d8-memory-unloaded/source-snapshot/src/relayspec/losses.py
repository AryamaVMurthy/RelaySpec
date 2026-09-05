from __future__ import annotations

import torch
import torch.nn.functional as F


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
        teacher_probabilities
        * (teacher_log_probabilities - student_log_probabilities)
    ).sum(dim=-1)
    return token_kl.mean() * (temperature**2)
