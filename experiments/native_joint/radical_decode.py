"""Inference-only hypotheses; every committed token still passes target verification."""
from types import SimpleNamespace
from functools import lru_cache

import torch
from torch.nn import functional as F
from transformers import DynamicCache


@lru_cache(maxsize=128)
def tree_layout(length, forks, lengths, device="cpu"):
    """Main path plus forked suffixes; a node attends only its own ancestors."""
    if length < 2 or len(lengths) != len(forks)+1 or lengths[0] != length or any(not 0 < p < n <= length for p, n in zip(forks, lengths[1:])):
        raise ValueError("Invalid branch fork")
    parents = [-1]+list(range(length-1))
    depths = list(range(length))
    paths = [list(range(length))]
    for fork, branch_length in zip(forks, lengths[1:]):
        begin = len(parents)
        parents.extend([fork-1]+list(range(begin, begin+branch_length-fork-1)))
        depths.extend(range(fork, branch_length))
        paths.append(list(range(fork))+list(range(begin, begin+branch_length-fork))+[0]*(length-branch_length))
    visible = [[False]*len(parents) for _ in parents]
    for node in range(len(parents)):
        parent = node
        while parent >= 0:
            visible[node][parent] = True
            parent = parents[parent]
    return (torch.tensor(depths, device=device), torch.tensor(paths, device=device),
            torch.tensor(visible, dtype=torch.bool, device=device), torch.tensor(lengths, device=device))


def pack_branches(proposal, forks, alternative_limit=None, leaves=False):
    if len(proposal) != len(forks)+1:
        raise ValueError("Each additional branch needs one fork")
    length = proposal.shape[1]
    lengths = (length,)+tuple(fork+1 if leaves else min(length, max(fork+1, alternative_limit or length)) for fork in forks)
    packed = torch.cat([proposal[0]]+[proposal[i, fork:lengths[i]] for i, fork in enumerate(forks, 1)]).unsqueeze(0)
    depths, paths, visible, lengths = tree_layout(length, tuple(forks), lengths, str(proposal.device))
    return packed, depths, paths, visible, lengths



def pack_adaptive_leaves(root, logits, budget=16, topk=5, prefix_weight=1.):
    """Fixed-size GPU selection among leaves sharing the primary prefix.

    Rank candidates by factorized primary-prefix mass times alternative mass.
    prefix_weight=0 ignores the prefix; this is a DDTree-inspired restricted
    candidate family, not a new general tree algorithm.
    """
    length = logits.shape[0]+1
    if not 0 < budget <= (length-1)*(topk-1):
        raise ValueError("Leaf budget exceeds candidate pool")
    top = logits.float().topk(topk, dim=-1)
    logp = top.values-logits.float().logsumexp(-1, keepdim=True)
    prefix = torch.cat([logp.new_zeros(1), logp[:-1, 0].cumsum(0)])
    scores = prefix_weight*prefix[:, None]+logp[:, 1:]
    chosen = scores.flatten().topk(budget).indices
    positions = torch.div(chosen, topk-1, rounding_mode='floor')+1
    ranks = chosen.remainder(topk-1)+1
    primary = torch.cat([root.reshape(1), logits.argmax(-1)])
    packed = torch.cat([primary, top.indices[positions-1, ranks]]).unsqueeze(0)
    nodes = torch.arange(length+budget, device=logits.device)
    visible = torch.zeros(length+budget, length+budget, dtype=torch.bool, device=logits.device)
    visible[:length, :length] = torch.ones(length, length, dtype=torch.bool, device=logits.device).tril()
    visible[length:] = (nodes[None] < positions[:, None]) | (nodes[None] == nodes[length:, None])
    depths = torch.cat([nodes[:length], positions])
    columns = torch.arange(length, device=logits.device)[None]
    leaf_paths = torch.where(columns < positions[:, None], columns, torch.where(columns == positions[:, None], nodes[length:, None], 0))
    paths = torch.cat([nodes[:length][None], leaf_paths])
    lengths = torch.cat([positions.new_tensor([length]), positions+1])
    return packed, depths, paths, visible, lengths, packed[0, paths]


def compact_tree_cache(cache, prefix_length, selected_nodes, main_path=False):
    """Keep the untouched prefix in place; copy only a selected fork's new nodes."""
    if not main_path:
        indices = prefix_length+selected_nodes
        end = prefix_length+len(selected_nodes)
        for layer in cache.layers:
            keys = layer.keys.index_select(-2, indices)
            values = layer.values.index_select(-2, indices)
            layer.keys[..., prefix_length:end, :].copy_(keys)
            layer.values[..., prefix_length:end, :].copy_(values)
    cache.crop(prefix_length+len(selected_nodes))


def lookup_continuation(history, suffix_length, maximum):
    """Retrieve already observed continuation tokens; never read future features."""
    if len(history) <= suffix_length:
        return []
    suffix = history[-suffix_length:]
    for start in range(len(history)-suffix_length-1, -1, -1):
        if history[start:start+suffix_length] == suffix:
            return history[start+suffix_length:start+suffix_length+maximum]
    return []


def warm_start_noise(noise, embed_tokens, carry, strength):
    """Blend only unverified proposal hints; retain the known first token exactly."""
    if not 0 <= strength <= 1:
        raise ValueError("Hint strength must be between zero and one")
    count = min(len(carry), noise.shape[1]-1)
    if count and strength:
        noise = noise.clone()
        hints = embed_tokens(torch.tensor([carry[:count]], device=noise.device))
        noise[:, 1:count+1] = torch.lerp(noise[:, 1:count+1], hints, strength)
    return noise


def rejected_tail(proposal, posterior, acceptance_length, source="draft"):
    """Hints after the new corrective token; they must be freshly verified."""
    if source == "draft":
        return proposal[acceptance_length+2:].tolist()
    if source == "target":
        # posterior[j] predicts the token after proposal[j]. Its first
        # rejection prediction becomes the known corrective token, not a hint.
        return posterior[acceptance_length+1:].tolist()
    raise ValueError("Unknown tail source")


def lazy_verify_tokens(proposal, first_logits, hidden, lm_head, chunk):
    """Compute full-vocabulary argmax only until the first rejection is known."""
    if proposal.shape[0] != 1 or chunk <= 0:
        raise ValueError("Lazy verification needs a single branch and positive chunk")
    posterior = first_logits.argmax(-1)
    while True:
        compared = min(posterior.shape[1], proposal.shape[1]-1)
        accepted = (proposal[:, 1:compared+1] == posterior[:, :compared]).cumprod(1).sum(1)
        if int(accepted[0]) < compared or posterior.shape[1] == proposal.shape[1]:
            return posterior, accepted
        begin = posterior.shape[1]
        extra = lm_head(hidden[:, begin:min(begin+chunk, proposal.shape[1])]).argmax(-1)
        posterior = torch.cat([posterior, extra], dim=1)


@torch.inference_mode()
def decode_variant(official, model, target, input_ids, max_new_tokens, stop_token_ids, variant, refiner=None, head=None):
    kind = variant["kind"]
    if kind not in {"fixed", "truncate", "confidence", "history", "branches", "tree_branches", "ddtree", "adaptive_leaves", "adaptive", "recycle", "shortlist", "window", "refine", "head", "warm_start", "lazy_head"}:
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
    draft_context_end = 0
    previous_logits = output.logits
    refine_cache = DynamicCache()
    if kind == "refine":
        if refiner is None or not 0 < variant["prefix"] < block_size-1:
            raise ValueError("Refinement needs a trained refiner and valid prefix")
        refine_pending = official.extract_context_feature(output.hidden_states, refiner.target_layer_ids)
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
        refined = False
        if history_proposal:
            proposal = torch.cat([output_ids[:, start:start+1], torch.tensor([history_proposal], device=target.device)], dim=1)
        else:
            noise = target.model.embed_tokens(proposal)
            if kind == "warm_start":
                noise = warm_start_noise(noise, target.model.embed_tokens, carry, variant["strength"])
            context_positions_begin = draft_cache.get_seq_length()
            if kind == "window":
                if target_hidden.shape[1] > variant["window"]:
                    target_hidden = target_hidden[:, -variant["window"]:]
                context_positions_begin = start-target_hidden.shape[1]
                if context_positions_begin < draft_context_end:
                    raise RuntimeError("Window context overlaps cached positions")
            draft_hidden = model(target_hidden=target_hidden, noise_embedding=noise, position_ids=position_ids[:, context_positions_begin:start+block_size], past_key_values=draft_cache, use_cache=True, is_causal=False)[:, 1-block_size:, :]
            if kind == "head":
                if head is None:
                    raise ValueError("A correction head is required")
                draft_hidden = head(draft_hidden)
            if kind == "window":
                # Cache contains absolute-position RoPE keys. Physical length can
                # shrink, while query positions always retain their true indices.
                draft_cache.crop(draft_cache.get_seq_length()-block_size)
                for layer in draft_cache.layers:
                    layer.keys = layer.keys[..., -variant["window"]:, :].contiguous()
                    layer.values = layer.values[..., -variant["window"]:, :].contiguous()
                draft_context_end = start
            else:
                draft_cache.crop(start)
            vocabulary = None
            if kind == "shortlist" and not first_draft:
                vocabulary = torch.unique(torch.cat([previous_logits[0].topk(variant["topk"], dim=-1).indices.flatten(), input_ids.flatten()]))
                draft_logits = F.linear(draft_hidden, target.lm_head.weight.index_select(0, vocabulary))
                proposal[:, 1:] = vocabulary[official.sample(draft_logits)]
            else:
                draft_logits = target.lm_head(draft_hidden)
                proposal[:, 1:] = official.sample(draft_logits)
            if kind == "tree_branches" and variant.get("main_length") is not None:
                if not 2 <= variant["main_length"] <= block_size:
                    raise ValueError("Main verification path must fit within the native draft block")
                proposal = proposal[:, :variant["main_length"]]
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
            elif kind in {"branches", "tree_branches"} and variant.get("branches", 2) > 1:
                top = draft_logits.float().topk(variant.get("draft_topk", 2), dim=-1)
                if variant.get("fork", "first") == "weakest":
                    search = min(variant.get("search_positions", 4), block_size-1)
                    fork_position = int((top.values[0, :search, 0]-top.values[0, :search, 1]).argmin())+1
                else:
                    fork_position = variant.get("fork_position", 1)
                if float(top.values[0, fork_position-1, 0]-top.values[0, fork_position-1, 1]) <= variant.get("branch_margin", float("inf")):
                    if variant.get("tree_style") == "leaves":
                        alternatives = [(p, rank) for p in range(1, min(proposal.shape[1], variant.get("leaf_positions", proposal.shape[1]-1)+1)) for rank in range(1, variant.get("draft_topk", 2))]
                    else:
                        alternatives = [(p, 1) for p in ([fork_position] if variant.get("branches", 2) == 2 else list(range(1, variant["branches"])))]
                    branch_forks = [p for p, _ in alternatives]
                    proposal = proposal.repeat(len(branch_forks)+1, 1)
                    for branch, (position, rank) in enumerate(alternatives, 1):
                        proposal[branch, position] = top.indices[0, position-1, rank]
                    if kind == "branches":
                        target_cache.batch_repeat_interleave(len(branch_forks)+1)
            elif kind == "refine":
                prefix = variant["prefix"]
                probabilities = (draft_logits[:, :prefix].float().amax(-1)-draft_logits[:, :prefix].float().logsumexp(-1)).exp()
                if float(probabilities.prod()) >= variant.get("threshold", 0.):
                    refinement_tokens = torch.full_like(proposal, model.mask_token_id)
                    refinement_tokens[:, :prefix+1] = proposal[:, :prefix+1]
                    refinement_hidden = refiner(target_hidden=refine_pending, noise_embedding=target.model.embed_tokens(refinement_tokens), position_ids=position_ids[:, refine_cache.get_seq_length():start+block_size], past_key_values=refine_cache, use_cache=True, is_causal=False)
                    proposal[:, prefix+1:] = target.lm_head(refinement_hidden[:, prefix+1:]).argmax(-1)
                    refine_cache.crop(start)
                    refined = True
        verify_length = proposal.shape[1]
        target_kwargs = {"logits_to_keep": slice(0, variant["chunk"])} if kind == "lazy_head" else {}
        tree_paths = None
        target_input = proposal
        target_positions = position_ids[:, start:start+verify_length].expand(proposal.shape[0], -1)
        if kind in {"tree_branches", "ddtree", "adaptive_leaves"}:
            if kind == "adaptive_leaves":
                target_input, depths, tree_paths, visible, tree_lengths, proposal = pack_adaptive_leaves(proposal[0, 0], draft_logits[0], variant.get("leaf_budget", 16), variant.get("draft_topk", 5), variant.get("prefix_weight", 1.))
                verify_length = proposal.shape[1]
            elif kind == "ddtree":
                from ddtree_baseline import pack_ddtree
                target_input, depths, tree_paths, visible, tree_lengths, proposal = pack_ddtree(proposal[0, 0], draft_logits[0].float()/variant.get("selection_temperature", 1.), variant["tree_budget"])
                verify_length = proposal.shape[1]
            else:
                target_input, depths, tree_paths, visible, tree_lengths = pack_branches(proposal, branch_forks, variant.get("branch_length"), variant.get("tree_style") == "leaves")
            target_positions = (depths+start).unsqueeze(0)
            allowed = torch.cat([torch.ones(len(depths), start, dtype=torch.bool, device=target.device), visible], dim=1)
            mask = torch.zeros(allowed.shape, dtype=target.dtype, device=target.device).masked_fill(~allowed, torch.finfo(target.dtype).min)
            target_kwargs["attention_mask"] = mask[None, None]
        output = target(target_input, position_ids=target_positions, past_key_values=target_cache, use_cache=True, output_hidden_states=True, **target_kwargs)
        previous_logits = output.logits
        if kind == "lazy_head":
            posterior, accepted = lazy_verify_tokens(proposal, output.logits, output.hidden_states[-1], target.lm_head, variant["chunk"])
        else:
            posterior = official.sample(output.logits, 0.)
            if tree_paths is not None:
                posterior = posterior[0, tree_paths]
            matches = proposal[:, 1:] == posterior[:, :-1]
            if tree_paths is not None:
                matches = matches & (torch.arange(verify_length-1, device=target.device)[None] < tree_lengths[:, None]-1)
            accepted = matches.cumprod(1).sum(1)
        winner = int(accepted.argmax())
        acceptance_length = int(accepted[winner])
        if branch_forks:
            for branch, position in enumerate(branch_forks, 1):
                if not torch.equal(posterior[0, :position], posterior[branch, :position]):
                    raise RuntimeError("Branches disagree before their different input token")
            if tree_paths is None:
                target_cache.batch_select_indices(torch.tensor([winner], device=target.device))
        if tree_paths is not None:
            compact_tree_cache(target_cache, start, tree_paths[winner, :acceptance_length+1], main_path=winner == 0 and kind != "ddtree")
        if kind in {"recycle", "warm_start"}:
            carry = rejected_tail(proposal[winner], posterior[winner], acceptance_length, variant.get("tail_source", "draft"))
            recycled_rounds = 0 if used_draft else recycled_rounds+1
        output_ids[:, start:start+acceptance_length+1] = proposal[winner:winner+1, :acceptance_length+1]
        output_ids[:, start+acceptance_length+1] = posterior[winner, acceptance_length]
        start += acceptance_length+1
        target_cache.crop(start)
        acceptance_lengths.append(acceptance_length+1)
        all_features = official.extract_context_feature(output.hidden_states, model.target_layer_ids)
        features = all_features[:, tree_paths[winner, :acceptance_length+1]] if tree_paths is not None else all_features[winner:winner+1, :acceptance_length+1]
        target_hidden = features if used_draft else torch.cat([target_hidden, features], dim=1)
        if kind == "refine":
            refine_features = official.extract_context_feature(output.hidden_states, refiner.target_layer_ids)[:, :acceptance_length+1]
            refine_pending = refine_features if refined else torch.cat([refine_pending, refine_features], dim=1)
        progress_ema = .7*progress_ema+.3*(acceptance_length+1)
        trace.append({"draft_call": bool(used_draft), "refined": refined, "verify_tokens": target_input.shape[1], "branches": target_input.shape[0], "proposal_branches": proposal.shape[0], "winner": winner, "progress": acceptance_length+1, "draft_context_tokens": draft_cache.get_seq_length(), "vocabulary": len(vocabulary) if used_draft and vocabulary is not None else target.lm_head.weight.shape[0]})
        trace[-1]["verification_head_tokens"] = output.logits.shape[1] if tree_paths is not None else posterior.shape[1]
        if stop_token_ids is not None and any(token in output_ids[:, num_input_tokens:] for token in stop_token_ids):
            break
    output_ids = output_ids[:, :min(start+1, max_length)]
    if stop_token_ids is not None:
        stops = torch.tensor(stop_token_ids, device=output_ids.device)
        found = torch.isin(output_ids[0, num_input_tokens:], stops).nonzero(as_tuple=True)[0]
        if found.numel():
            output_ids = output_ids[:, :num_input_tokens+found[0]+1]
    elapsed = official._cuda_time()-decode_start
    return SimpleNamespace(output_ids=output_ids, num_output_tokens=output_ids.shape[1]-num_input_tokens, acceptance_lengths=acceptance_lengths, trace=trace, decode_seconds=elapsed)
