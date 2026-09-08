"""Inference-only hypotheses; every committed token still passes target verification."""
from types import SimpleNamespace

import torch
from transformers import DynamicCache


def lookup_continuation(history, suffix_length, maximum):
    """Retrieve already observed continuation tokens; never read future features."""
    if len(history) <= suffix_length:
        return []
    suffix = history[-suffix_length:]
    for start in range(len(history)-suffix_length-1, -1, -1):
        if history[start:start+suffix_length] == suffix:
            return history[start+suffix_length:start+suffix_length+maximum]
    return []


@torch.inference_mode()
def decode_variant(official, model, target, input_ids, max_new_tokens, stop_token_ids, variant):
    kind = variant["kind"]
    if kind not in {"fixed", "truncate", "confidence", "history", "branches", "adaptive", "recycle"}:
        raise ValueError("Unknown decoding hypothesis")
    block_size = variant.get("block_size", 16)
    max_block = max(block_size, variant.get("lookup_block", 16), 32 if kind == "adaptive" else 1)
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens+max_new_tokens
    output_ids = torch.full((1, max_length+max_block), model.mask_token_id, dtype=torch.long, device=target.device)
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    target_cache, draft_cache = DynamicCache(), DynamicCache()
    official._cuda_time()
    output = target(input_ids, position_ids=position_ids[:, :num_input_tokens], past_key_values=target_cache, use_cache=True, logits_to_keep=1, output_hidden_states=True)
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens+1] = official.sample(output.logits, 0.)
    target_hidden = official.extract_context_feature(output.hidden_states, model.target_layer_ids)
    official._cuda_time(); decode_start = official._cuda_time()
    acceptance_lengths, trace = [], []
    start, first_draft, progress_ema = num_input_tokens, True, 7.
    carry, recycled_rounds = [], 0
    while start < max_length:
        if kind == "adaptive":
            choices = variant.get("choices", [4, 8, 12, 16, 24, 32])
            wanted = progress_ema*variant.get("headroom", 1.5)
            block_size = min(choices, key=lambda size: abs(size-wanted))
        proposal = output_ids[:, start:start+block_size].clone()
        history_proposal = []
        if kind == "recycle" and recycled_rounds < variant.get("max_recycles", 1) and len(carry) >= variant.get("minimum_recycle", 4):
            history_proposal = carry
        if kind == "history":
            history = output_ids[0, :start+1].tolist()
            history_proposal = lookup_continuation(history, variant.get("suffix", 3), variant.get("lookup_block", 16)-1)
            if len(history_proposal) < variant.get("minimum_lookup", 3):
                history_proposal = []
        used_draft = not history_proposal
        branch_forks = []
        if history_proposal:
            proposal = torch.cat([output_ids[:, start:start+1], torch.tensor([history_proposal], device=target.device)], dim=1)
        else:
            noise = target.model.embed_tokens(proposal)
            draft_logits = target.lm_head(model(target_hidden=target_hidden, noise_embedding=noise, position_ids=position_ids[:, draft_cache.get_seq_length():start+block_size], past_key_values=draft_cache, use_cache=True, is_causal=False)[:, 1-block_size:, :])
            draft_cache.crop(start)
            proposal[:, 1:] = official.sample(draft_logits)
            if first_draft:
                first_draft = False
                decode_start = official._cuda_time()
            if kind == "truncate":
                proposal = proposal[:, :variant["verify_tokens"]]
            elif kind == "confidence":
                confidence = (draft_logits.float().amax(-1)-draft_logits.float().logsumexp(-1)).exp()[0]
                if variant.get("cumulative"):
                    confidence = confidence.cumprod(0)
                keep = int((confidence >= variant["threshold"]).cumprod(0).sum())+1
                proposal = proposal[:, :max(variant.get("minimum", 2), keep)]
            elif kind == "branches":
                top = draft_logits.float().topk(2, dim=-1)
                if variant.get("fork", "first") == "weakest":
                    search = min(variant.get("search_positions", 4), block_size-1)
                    fork_position = int((top.values[0, :search, 0]-top.values[0, :search, 1]).argmin())+1
                else:
                    fork_position = variant.get("fork_position", 1)
                if float(top.values[0, fork_position-1, 0]-top.values[0, fork_position-1, 1]) <= variant.get("branch_margin", float("inf")):
                    branch_forks = [fork_position] if variant.get("branches", 2) == 2 else list(range(1, variant["branches"]))
                    proposal = proposal.repeat(len(branch_forks)+1, 1)
                    for branch, position in enumerate(branch_forks, 1):
                        proposal[branch, position] = top.indices[0, position-1, 1]
                    target_cache.batch_repeat_interleave(len(branch_forks)+1)
        verify_length = proposal.shape[1]
        output = target(proposal, position_ids=position_ids[:, start:start+verify_length].expand(proposal.shape[0], -1), past_key_values=target_cache, use_cache=True, output_hidden_states=True)
        posterior = official.sample(output.logits, 0.)
        accepted = (proposal[:, 1:] == posterior[:, :-1]).cumprod(1).sum(1)
        winner = int(accepted.argmax())
        acceptance_length = int(accepted[winner])
        if branch_forks:
            for branch, position in enumerate(branch_forks, 1):
                if not torch.equal(posterior[0, :position], posterior[branch, :position]):
                    raise RuntimeError("Branches disagree before their different input token")
            target_cache.batch_select_indices(torch.tensor([winner], device=target.device))
        if kind == "recycle":
            carry = proposal[winner, acceptance_length+2:].tolist()
            recycled_rounds = 0 if used_draft else recycled_rounds+1
        output_ids[:, start:start+acceptance_length+1] = proposal[winner:winner+1, :acceptance_length+1]
        output_ids[:, start+acceptance_length+1] = posterior[winner, acceptance_length]
        start += acceptance_length+1
        target_cache.crop(start)
        acceptance_lengths.append(acceptance_length+1)
        features = official.extract_context_feature(output.hidden_states, model.target_layer_ids)[winner:winner+1, :acceptance_length+1]
        target_hidden = features if used_draft else torch.cat([target_hidden, features], dim=1)
        progress_ema = .7*progress_ema+.3*(acceptance_length+1)
        trace.append({"draft_call": bool(used_draft), "verify_tokens": verify_length, "branches": proposal.shape[0], "winner": winner, "progress": acceptance_length+1})
        if stop_token_ids is not None and any(token in output_ids[:, num_input_tokens:] for token in stop_token_ids):
            break
    output_ids = output_ids[:, :min(start+1, max_length)]
    if stop_token_ids is not None:
        stops = torch.tensor(stop_token_ids, device=output_ids.device)
        found = torch.isin(output_ids[0, num_input_tokens:], stops).nonzero(as_tuple=True)[0]
        if found.numel():
            output_ids = output_ids[:, :num_input_tokens+found[0]+1]
    elapsed = official._cuda_time()-decode_start
    return SimpleNamespace(output_ids=output_ids, acceptance_lengths=acceptance_lengths, trace=trace, decode_seconds=elapsed)
