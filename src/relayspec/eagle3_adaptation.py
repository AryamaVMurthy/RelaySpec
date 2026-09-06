"""Teacher-forced one-step EAGLE supervision, separate from multi-step TTT training."""

import torch


def shifted_eagle_inputs(features, input_ids, *, block=16):
    """Feature[t] + token[t+1] predicts token[t+2]; labels come from target[t+1]."""
    if (
        features.ndim != 3
        or input_ids.ndim != 2
        or features.shape[:2] != input_ids.shape
        or input_ids.shape[1] <= block
    ):
        raise ValueError(
            "EAGLE supervision needs aligned unpadded features and a full label block"
        )
    positions = torch.arange(input_ids.shape[1] - 2, device=input_ids.device).unsqueeze(
        0
    )
    return features[:, :-2], input_ids[:, 1:-1], positions


def eagle_teacher_forced_logits(
    draft, mapper, features, tokens, positions, *, label_count=15
):
    if (
        features.shape[1] < label_count
        or tokens.shape != positions.shape
        or features.shape[:2] != tokens.shape
    ):
        raise ValueError("EAGLE shifted context and token positions must align")
    hidden = draft(
        hidden_states=mapper(features),
        input_ids=tokens,
        position_ids=positions,
        past_key_values=None,
        use_cache=False,
    )
    return draft.compute_logits(hidden[:, -label_count:])


@torch.no_grad()
def check_eagle_initial_cache(draft, mapper, record, *, cache_factory):
    """Full-prefix training forward and the released inference prefix must agree."""
    features, tokens, positions = record[:3]
    cache = cache_factory()
    hidden = draft.extend_draft_cache(
        hidden_states=mapper(features),
        input_ids=tokens,
        position_ids=positions,
        past_key_values=cache,
    )
    cached = draft.compute_logits(hidden)
    direct = eagle_teacher_forced_logits(
        draft, mapper, features, tokens, positions, label_count=1
    )
    if cache.get_seq_length() != tokens.shape[1] or not torch.equal(cached, direct):
        raise ValueError(
            "EAGLE teacher-forced prefix differs from released cache initialization"
        )
    return {
        "status": "pass",
        "prefix_tokens": tokens.shape[1],
        "logits_bit_identical": True,
        "scope": "One-step teacher-forced alignment and initial-cache equivalence for this prefix, not multi-step TTT or full-generation quality.",
    }
