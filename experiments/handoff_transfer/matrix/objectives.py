from __future__ import annotations
"""AUF formula implementation atop pinned SpecForge (not author-released code).
Decay uses the upstream implementation without changes. No new model inputs.
"""
import types
import torch
from torch.nn import functional as F

def configure(model,objective):
    model._matrix_objective=objective
    if objective=='decay':
        model.loss_decay_gamma={16:7.,10:5.,8:4.}[model.block_size]
    elif objective in ('auf','ce'):
        global DFlashObjectiveTerms,SelectorTerms
        from specforge.algorithms.common.dflash_family_model import DFlashObjectiveTerms,SelectorTerms
        model.loss_decay_gamma=None
        model._dflash_objective_chunk_terms=types.MethodType(_dflash_objective_chunk_terms,model)
    else:raise ValueError(objective)

def _dflash_objective_chunk_terms(
    self,
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    weight_mask: torch.Tensor,
    predecessor_ids: torch.Tensor,
) -> DFlashObjectiveTerms:
    """Return a flat tuple of additive objective and metric tensors."""

    batch_size, num_blocks, block_size, hidden_size = hidden.shape
    logits = self.lm_head(
        hidden.reshape(batch_size, num_blocks * block_size, hidden_size)
    ).reshape(batch_size, num_blocks, block_size, -1)
    candidate_selector = getattr(self.draft_model, "candidate_selector", None)
    objective_logits = (
        self.draft_model.transform_unary_logits(logits)
        if candidate_selector is not None
        else logits
    )
    neg_log_q = F.cross_entropy(
        objective_logits.reshape(-1, objective_logits.shape[-1]),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape_as(target_ids)

    target_probability = torch.exp(-neg_log_q)
    # Argmax is shared by AUF support and accuracy: one vocabulary reduction.
    with torch.no_grad():
        predicted_ids = objective_logits.argmax(dim=-1)
    if self._matrix_objective == 'auf':
        # Detached strict-prefix support: keep first failure, discard later labels.
        with torch.no_grad():
            valid = weight_mask > 0
            correct = (predicted_ids == target_ids) | ~valid
            before = torch.cat([torch.ones_like(correct[..., :1]), correct[..., :-1]], dim=-1)
            support = before.to(torch.int64).cumprod(-1).to(weight_mask.dtype)
        loss_weights = weight_mask * support
    else:
        loss_weights = weight_mask
    loss_den=loss_weights.sum()

    ce_loss_num = (neg_log_q * loss_weights).sum()
    if self.lk_loss_type in {"lambda", "tv"}:
        tv_loss_num = ((1.0 - target_probability) * loss_weights).sum()
    else:
        tv_loss_num = ce_loss_num.new_zeros(())
    target_probability_num = (target_probability.detach() * weight_mask).sum()

    selector_terms = SelectorTerms.zeros(ce_loss_num)
    if self._selector_objective_enabled:
        selector_terms = self._selector_chunk_terms(
            candidate_selector=candidate_selector,
            objective_logits=objective_logits,
            hidden=hidden,
            target_ids=target_ids,
            predecessor_ids=predecessor_ids,
            loss_weights=loss_weights,
            weight_mask=weight_mask,
        )

    with torch.no_grad():
        correct_num = (
            ((predicted_ids == target_ids) & (weight_mask > 0.5)).sum().float()
        )
        accuracy_den = weight_mask.sum()
    return DFlashObjectiveTerms(
        ce_loss_num=ce_loss_num,
        tv_loss_num=tv_loss_num,
        loss_den=loss_den,
        target_probability_num=target_probability_num,
        correct_num=correct_num,
        accuracy_den=accuracy_den,
        selector_ce_num=selector_terms.ce_num,
        selector_probability_num=selector_terms.probability_num,
        selector_correct_num=selector_terms.correct_num,
        selector_weight_den=selector_terms.weight_den,
        selector_covered_num=selector_terms.covered_num,
    )
