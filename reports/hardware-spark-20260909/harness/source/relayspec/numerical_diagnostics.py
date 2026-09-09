"""Explicit precision interventions for diagnosing greedy-output agreement."""

import copy

import torch
from torch import nn


def _float32_head_inputs(_module, args):
    if len(args) != 1:
        raise ValueError("Output-head diagnostic expects one hidden-state tensor")
    return (args[0].to(torch.float32),)


def promote_output_head(model: nn.Module) -> dict:
    """Compute only the output projection in FP32, preserving tied embeddings."""
    original = model.get_output_embeddings()
    embedding = model.get_input_embeddings().weight
    embedding_dtype = embedding.dtype
    if not isinstance(original, nn.Linear) or original.weight.dtype != torch.bfloat16:
        raise ValueError("Head-only diagnostic requires a BF16 linear output head")
    tied = original.weight is embedding
    # Casting a tied head in place would also cast the input embedding and break
    # the BF16 transformer. Copy the head before the precision intervention.
    promoted = copy.deepcopy(original).to(dtype=torch.float32)
    promoted.register_forward_pre_hook(_float32_head_inputs)
    model.set_output_embeddings(promoted)
    if model.get_input_embeddings().weight is not embedding or embedding.dtype != embedding_dtype:
        raise ValueError("Output-head promotion changed input embeddings")
    return dict(originally_tied=tied, output_head_dtype=str(promoted.weight.dtype),
                input_embedding_dtype=str(embedding.dtype),
                parameters=promoted.weight.numel(), copied_head=True)
