from __future__ import annotations

import math

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
        cosine = (
            1 - F.cosine_similarity(prediction.float(), target.float(), dim=-1).mean()
        )
        return mse + float(historical_cosine_weight) * cosine
    raise ValueError(f"unknown interface objective: {objective}")


def _check_verifier_shapes(
    proposal_logits: torch.Tensor,
    accepted_tokens: torch.Tensor,
) -> None:
    if proposal_logits.ndim != 3:
        raise ValueError(
            "proposal logits must have shape [batch, positions, vocabulary]"
        )
    if accepted_tokens.ndim != 2:
        raise ValueError("verifier tokens must have shape [batch, positions]")
    if proposal_logits.shape[:2] != accepted_tokens.shape:
        raise ValueError(
            "proposal logits and verifier tokens must agree on batch and positions"
        )


def greedy_agreement_ce(
    proposal_logits: torch.Tensor,
    accepted_tokens: torch.Tensor,
    *,
    position_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy against the token the target verifier would commit.

    This supervises the relay with the target's own greedy choice instead of a
    reconstructed source tensor, so no source trunk forward pass is required.
    """
    _check_verifier_shapes(proposal_logits, accepted_tokens)
    log_probabilities = F.log_softmax(proposal_logits.float(), dim=-1)
    token_log_probability = log_probabilities.gather(
        -1,
        accepted_tokens.detach().unsqueeze(-1),
    ).squeeze(-1)
    negative_log_likelihood = -token_log_probability
    if position_weights is None:
        return negative_log_likelihood.mean()
    if position_weights.shape != accepted_tokens.shape[-1:]:
        raise ValueError("position weights must have shape [positions]")
    weights = position_weights.detach().float()
    weight_total = weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
    return (negative_log_likelihood * weights).sum(dim=-1).mean() / weight_total


def expected_accepted_length_surrogate(
    proposal_logits: torch.Tensor,
    accepted_tokens: torch.Tensor,
) -> torch.Tensor:
    """Differentiable accepted-prefix surrogate ``A_soft``, not the deployed
    quantity ``A_hard``.

    Writing ``a_j`` for the proposer's probability mass on the token the
    verifier would commit at position ``j``, this returns the negative of

        A_soft = sum_i prod_{j<=i} a_j

    ``A_soft`` is the *exact* expected accepted prefix length only under a
    sampled-proposal regime, where each draft token is drawn from the
    proposer's distribution and accepted iff it equals the verifier's token.
    The regime this codebase actually deploys is greedy argmax, whose
    accepted length is

        A_hard = sum_i prod_{j<=i} 1[argmax_v q_D(v | .) == accepted_tokens[j]]

    which is a deterministic, non-differentiable function of the logits.
    **No general inequality holds between A_soft and A_hard in either
    direction**: a diffuse distribution can have a_j < 0.5 while the
    verifier's token is still the argmax (A_soft undercounts that position),
    and a near-tie can have a_j around 0.45 lose the argmax to a competitor
    at 0.5 (A_soft overcounts). The one guaranteed implication is
    ``a_j > 1/2 => position j is accepted under greedy``, which is the
    mechanism by which optimizing A_soft is expected to move A_hard.

    Because no equivalence holds, this surrogate is licensed empirically, not
    analytically: callers must log the hard accepted-prefix length (see
    ``relayspec.metrics``) alongside this value and verify that improving
    A_soft on held-out data actually improves A_hard. This function computes
    A_soft only.

    The returned value is ``-A_soft`` normalized by the number of proposed
    positions, so it lies in ``[-1, 0]`` and is comparable across block sizes.
    """
    _check_verifier_shapes(proposal_logits, accepted_tokens)
    positions = proposal_logits.shape[1]
    if positions < 1:
        raise ValueError("expected accepted length needs at least one position")
    log_probabilities = F.log_softmax(proposal_logits.float(), dim=-1)
    token_log_probability = log_probabilities.gather(
        -1,
        accepted_tokens.detach().unsqueeze(-1),
    ).squeeze(-1)
    prefix_log_probability = token_log_probability.cumsum(dim=-1)
    expected_length = prefix_log_probability.exp().sum(dim=-1)
    return -(expected_length / positions).mean()


def hard_accepted_prefix_diagnostic(
    proposal_logits: torch.Tensor,
    accepted_tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Deployed-regime accept diagnostic, computed with no gradient.

    Returns the hard accepted-prefix length ``A_hard`` per example (the
    greedy accept rule this codebase actually deploys: the longest prefix of
    positions whose argmax equals the verifier's token), the per-position
    hard-acceptance indicator, and the mean per-position agreement mass
    ``a_j``. This is the quantity that licenses ``expected_accepted_length_surrogate``
    empirically: every fit using that surrogate must log this diagnostic
    alongside it and confirm the two move together.
    """
    _check_verifier_shapes(proposal_logits, accepted_tokens)
    with torch.no_grad():
        argmax_tokens = proposal_logits.argmax(dim=-1)
        agreed = argmax_tokens == accepted_tokens
        # A position counts only if every earlier position also agreed, so a
        # cumulative product of the agreement booleans is exactly the
        # leading-run indicator, and its sum is the accepted prefix length.
        leading_run = agreed.long().cumprod(dim=-1)
        hard_accepted_length = leading_run.sum(dim=-1).float()
        log_probabilities = F.log_softmax(proposal_logits.float(), dim=-1)
        agreement_mass = (
            log_probabilities.gather(
                -1,
                accepted_tokens.unsqueeze(-1),
            )
            .squeeze(-1)
            .exp()
        )
        return {
            "hard_accepted_length": hard_accepted_length,
            "hard_accepted_indicator": agreed,
            "agreement_mass": agreement_mass,
        }


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


def explicit_l2_penalty(parameters, coefficient: float) -> torch.Tensor:
    """lambda/2 * sum(w**2) over trainable parameters, distinct from AdamW decay."""
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("L2 coefficient must be finite and non-negative")
    weights = [p for p in parameters if p.requires_grad]
    if not weights:
        raise ValueError("L2 requires at least one trainable parameter")
    if coefficient == 0:
        return weights[0].new_zeros((), dtype=torch.float32)
    return (coefficient / 2) * sum(p.float().square().sum() for p in weights)
