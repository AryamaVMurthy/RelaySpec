from __future__ import annotations

import time
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import torch

from relayspec.proposers import SourceContextProvider
from relayspec.relay import TargetFeatureRelay, extract_hidden_taps
from relayspec.source import SourceTapProvider


def _profile_region(recorder: Any | None, name: str) -> Any:
    return recorder.region(name) if recorder is not None else nullcontext()


def greedy_sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return logits.argmax(dim=-1)
    batch, sequence, vocabulary = logits.shape
    probabilities = torch.softmax(logits.reshape(-1, vocabulary) / temperature, dim=-1)
    return torch.multinomial(probabilities, num_samples=1).reshape(batch, sequence)


def accepted_block_length(block_ids: torch.Tensor, posterior: torch.Tensor) -> int:
    if block_ids.ndim != 2 or posterior.shape != block_ids.shape:
        raise ValueError("block_ids and posterior must have the same [batch, block] shape")
    if block_ids.shape[0] != 1 or block_ids.shape[1] < 2:
        raise ValueError("compiled DFlash currently requires batch one and block length >= 2")
    matches = block_ids[:, 1:].eq(posterior[:, :-1])
    accepted_drafts = matches.to(torch.int64).cumprod(dim=1).sum(dim=1)[0].item()
    return int(accepted_drafts) + 1


@torch.inference_mode()
def native_autoregressive_generate(
    target: torch.nn.Module,
    *,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: list[int] | None,
    temperature: float,
    return_stats: bool = False,
) -> Any:
    """Generate directly from the native target while retaining comparable timings."""
    if input_ids.shape[0] != 1 or max_new_tokens <= 0:
        raise ValueError("batch size one and a positive generation length are required")
    target.eval()
    torch.cuda.synchronize()
    prefill_started = time.perf_counter()
    prefill = target(input_ids, use_cache=True, logits_to_keep=1)
    next_token = greedy_sample(prefill.logits, temperature)
    cache = getattr(prefill, "past_key_values", None)
    output_ids = torch.cat((input_ids, next_token), dim=1)
    torch.cuda.synchronize()
    time_to_first_token = time.perf_counter() - prefill_started
    target_calls = 1
    stopped = stop_token_ids is not None and int(next_token.item()) in stop_token_ids

    torch.cuda.synchronize()
    decode_started = time.perf_counter()
    while output_ids.shape[1] - input_ids.shape[1] < max_new_tokens and not stopped:
        output = target(
            output_ids[:, -1:],
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = getattr(output, "past_key_values", cache)
        next_token = greedy_sample(output.logits, temperature)
        output_ids = torch.cat((output_ids, next_token), dim=1)
        target_calls += 1
        stopped = stop_token_ids is not None and int(next_token.item()) in stop_token_ids
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_started
    if not return_stats:
        return output_ids
    num_output_tokens = output_ids.shape[1] - input_ids.shape[1]
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=input_ids.shape[1],
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=decode_seconds / max(num_output_tokens, 1),
        acceptance_lengths=[1] * num_output_tokens,
        target_calls=target_calls,
        draft_calls=0,
    )


@torch.inference_mode()
def matched_full_target_dflash_generate(
    draft: torch.nn.Module,
    *,
    source_provider: SourceTapProvider,
    native_target: torch.nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: list[int] | None,
    temperature: float,
    block_size: int | None = None,
    mask_token_id: int | None = None,
    return_stats: bool = False,
    profile_recorder: Any | None = None,
) -> Any:
    """Run the matched source-trunk DFlash baseline with full verification."""
    if input_ids.shape[0] != 1 or max_new_tokens <= 0:
        raise ValueError("batch size one and a positive generation length are required")
    if tuple(draft.target_layer_ids) != source_provider.tap_layers:
        raise ValueError(
            "source-provider taps must exactly match the trained DFlash context layers"
        )
    block_size = draft.block_size if block_size is None else block_size
    mask_token_id = draft.mask_token_id if mask_token_id is None else mask_token_id
    if block_size < 2 or mask_token_id is None:
        raise ValueError("a valid speculative block and mask token are required")

    draft.eval()
    source_provider.eval()
    native_target.eval()
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    device = input_ids.device
    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    from transformers import DynamicCache

    target_cache = DynamicCache(config=native_target.config)
    source_context_provider = SourceContextProvider(
        tap_provider=source_provider,
        source_cache=DynamicCache(config=source_provider.source.config),
        project=lambda features: features,
    )
    draft_cache = DynamicCache()

    torch.cuda.synchronize()
    prefill_started = time.perf_counter()
    with _profile_region(profile_recorder, "prefill_source_trunk"):
        proposer_context = source_context_provider.initialize(
            input_ids=input_ids,
            target_output=None,
        )
    with _profile_region(profile_recorder, "prefill_full_target"):
        target_prefill = native_target(
            input_ids,
            past_key_values=target_cache,
            use_cache=True,
            logits_to_keep=1,
        )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = greedy_sample(
        target_prefill.logits,
        temperature,
    )
    torch.cuda.synchronize()
    time_to_first_token = time.perf_counter() - prefill_started

    start = num_input_tokens
    acceptance_lengths: list[int] = []
    target_calls = 0
    draft_calls = 0
    torch.cuda.synchronize()
    decode_started = time.perf_counter()
    while start < max_length - 1:
        with _profile_region(profile_recorder, "draft"):
            block_ids = output_ids[:, start : start + block_size].clone()
            noise_embedding = source_provider.source.model.embed_tokens(block_ids)
            draft_hidden = draft(
                target_hidden=proposer_context,
                noise_embedding=noise_embedding,
                position_ids=position_ids[
                    :, draft_cache.get_seq_length() : start + block_size
                ],
                past_key_values=draft_cache,
                use_cache=True,
                is_causal=False,
            )
            draft_calls += 1
            draft_logits = source_provider.source.lm_head(
                draft_hidden[:, 1 - block_size :, :]
            )
            draft_cache.crop(start)
            block_ids[:, 1:] = greedy_sample(draft_logits, temperature)

        with _profile_region(profile_recorder, "verification_full_target"):
            target_verification = native_target(
                block_ids,
                past_key_values=target_cache,
                use_cache=True,
                logits_to_keep=block_size,
            )
        target_calls += 1
        posterior = greedy_sample(target_verification.logits, temperature)
        accepted = accepted_block_length(block_ids, posterior)
        output_ids[:, start : start + accepted] = block_ids[:, :accepted]
        output_ids[:, start + accepted] = posterior[:, accepted - 1]
        target_cache.crop(start + accepted)
        with _profile_region(profile_recorder, "verification_source_trunk"):
            proposer_context = source_context_provider.update(
                verification_input_ids=block_ids,
                target_output=target_verification,
                committed_length=accepted,
            )
        start += accepted
        acceptance_lengths.append(accepted)

        if stop_token_ids is not None:
            generated = output_ids[:, num_input_tokens : start + 1]
            stops = torch.tensor(stop_token_ids, device=device)
            if torch.isin(generated, stops).any().item():
                break

    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_started
    output_ids = output_ids[:, : min(start + 1, max_length)]
    if stop_token_ids is not None:
        stops = torch.tensor(stop_token_ids, device=device)
        indices = torch.isin(output_ids[0, num_input_tokens:], stops).nonzero(
            as_tuple=True
        )[0]
        if indices.numel():
            output_ids = output_ids[:, : num_input_tokens + int(indices[0].item()) + 1]
    if not return_stats:
        return output_ids
    num_output_tokens = output_ids.shape[1] - num_input_tokens
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=decode_seconds / max(num_output_tokens, 1),
        acceptance_lengths=acceptance_lengths or [1],
        target_calls=target_calls,
        draft_calls=draft_calls,
        source_executed_tokens=source_context_provider.executed_tokens,
    )


def _conditioned_dflash_forward(
    draft: torch.nn.Module,
    *,
    conditioned_context: torch.Tensor,
    noise_embedding: torch.Tensor,
    position_ids: torch.LongTensor,
    past_key_values: Any,
) -> torch.Tensor:
    """Run DFlash after its source-feature projection has been replaced by a relay."""
    hidden_states = noise_embedding
    position_embeddings = draft.rotary_emb(hidden_states, position_ids)
    for layer in draft.layers:
        hidden_states = layer(
            hidden_states=hidden_states,
            target_hidden=conditioned_context,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=True,
            position_embeddings=position_embeddings,
            is_causal=False,
        )
    return draft.norm(hidden_states)


@torch.inference_mode()
def relay_dflash_generate(
    draft: torch.nn.Module,
    *,
    relay: TargetFeatureRelay,
    relay_target_layer_ids: tuple[int, ...],
    native_target: torch.nn.Module,
    source_embedding: torch.nn.Module,
    source_lm_head: torch.nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: list[int] | None,
    temperature: float,
    block_size: int | None = None,
    mask_token_id: int | None = None,
    return_stats: bool = False,
    profile_recorder: Any | None = None,
) -> Any:
    """Run verified DFlash while relaying cached target taps to the proposer.

    The native target still verifies every proposed block. The relay changes only
    proposal quality under ordinary speculative acceptance. Byte equality with a
    one-token loop is backend-dependent because block-shaped kernels can differ
    numerically near greedy ties.
    """
    if input_ids.shape[0] != 1 or max_new_tokens <= 0:
        raise ValueError("batch size one and a positive generation length are required")
    block_size = draft.block_size if block_size is None else block_size
    mask_token_id = draft.mask_token_id if mask_token_id is None else mask_token_id
    if block_size < 2 or mask_token_id is None:
        raise ValueError("a valid speculative block and mask token are required")
    if len(relay_target_layer_ids) != len(draft.target_layer_ids):
        raise ValueError("relay and trained DFlash must use the same number of taps")

    draft.eval()
    relay.eval()
    native_target.eval()
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    device = input_ids.device
    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    from transformers import DynamicCache

    target_cache = DynamicCache(config=native_target.config)
    draft_cache = DynamicCache()

    torch.cuda.synchronize()
    prefill_started = time.perf_counter()
    with _profile_region(profile_recorder, "prefill_full_target"):
        target_prefill = native_target(
            input_ids,
            past_key_values=target_cache,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = greedy_sample(
        target_prefill.logits,
        temperature,
    )
    with _profile_region(profile_recorder, "prefill_relay"):
        relay_features = extract_hidden_taps(
            target_prefill.hidden_states,
            relay_target_layer_ids,
        )
        conditioned_context = draft.hidden_norm(relay(relay_features))
    torch.cuda.synchronize()
    time_to_first_token = time.perf_counter() - prefill_started

    start = num_input_tokens
    acceptance_lengths: list[int] = []
    target_calls = 0
    draft_calls = 0
    torch.cuda.synchronize()
    decode_started = time.perf_counter()
    while start < max_length - 1:
        with _profile_region(profile_recorder, "draft"):
            block_ids = output_ids[:, start : start + block_size].clone()
            noise_embedding = source_embedding(block_ids)
            draft_hidden = _conditioned_dflash_forward(
                draft,
                conditioned_context=conditioned_context,
                noise_embedding=noise_embedding,
                position_ids=position_ids[
                    :, draft_cache.get_seq_length() : start + block_size
                ],
                past_key_values=draft_cache,
            )
            draft_calls += 1
            draft_logits = source_lm_head(draft_hidden[:, 1 - block_size :, :])
            draft_cache.crop(start)
            block_ids[:, 1:] = greedy_sample(draft_logits, temperature)

        with _profile_region(profile_recorder, "verification_full_target"):
            target_verification = native_target(
                block_ids,
                past_key_values=target_cache,
                use_cache=True,
                output_hidden_states=True,
            )
        target_calls += 1
        posterior = greedy_sample(target_verification.logits, temperature)
        accepted = accepted_block_length(block_ids, posterior)
        output_ids[:, start : start + accepted] = block_ids[:, :accepted]
        output_ids[:, start + accepted] = posterior[:, accepted - 1]
        target_cache.crop(start + accepted)
        start += accepted
        acceptance_lengths.append(accepted)
        with _profile_region(profile_recorder, "relay"):
            relay_features = extract_hidden_taps(
                target_verification.hidden_states,
                relay_target_layer_ids,
            )[:, :accepted, :]
            conditioned_context = draft.hidden_norm(relay(relay_features))

        if stop_token_ids is not None:
            generated = output_ids[:, num_input_tokens : start + 1]
            stops = torch.tensor(stop_token_ids, device=device)
            if torch.isin(generated, stops).any().item():
                break

    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_started
    output_ids = output_ids[:, : min(start + 1, max_length)]
    if stop_token_ids is not None:
        stops = torch.tensor(stop_token_ids, device=device)
        indices = torch.isin(output_ids[0, num_input_tokens:], stops).nonzero(
            as_tuple=True
        )[0]
        if indices.numel():
            output_ids = output_ids[:, : num_input_tokens + int(indices[0].item()) + 1]
    if not return_stats:
        return output_ids
    num_output_tokens = output_ids.shape[1] - num_input_tokens
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=decode_seconds / max(num_output_tokens, 1),
        acceptance_lengths=acceptance_lengths or [1],
        target_calls=target_calls,
        draft_calls=draft_calls,
    )
