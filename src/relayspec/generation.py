from __future__ import annotations

import time
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import torch

from relayspec.proposers import SourceContextProvider
from relayspec.relay import TargetFeatureRelay, extract_hidden_taps
from relayspec.source import SourceTapProvider
from relayspec.vocab_bridge import (
    bridge_encode_new_chunk,
    committed_suffix,
    decode_new_chunk,
    encode_new_chunk,
)


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
        raise ValueError(
            "block_ids and posterior must have the same [batch, block] shape"
        )
    if block_ids.shape[0] != 1 or block_ids.shape[1] < 2:
        raise ValueError(
            "compiled DFlash currently requires batch one and block length >= 2"
        )
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
        stopped = (
            stop_token_ids is not None and int(next_token.item()) in stop_token_ids
        )
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
    proposal_lengths: list[int] = []
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
            # Match released DFlash: proposals are greedy even when target tokens sample.
            block_ids[:, 1:] = greedy_sample(draft_logits)

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
        proposal_lengths.append(int(block_ids.shape[1]) - 1)

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
        proposal_lengths=proposal_lengths,
        accepted_draft_lengths=[x - 1 for x in acceptance_lengths],
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
    selective_capture: bool = False,
    release_capture_buffers: bool = False,
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
    if not relay_target_layer_ids:
        raise ValueError("relay must condition on at least one target tap")

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
        if selective_capture:
            from relayspec.selected_taps import forward_selected_taps

            target_prefill, selected_prefill_taps = forward_selected_taps(
                native_target, input_ids, relay_target_layer_ids,
                past_key_values=target_cache, use_cache=True, logits_to_keep=1,
            )
        else:
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
        if selective_capture:
            relay_features = torch.cat(selected_prefill_taps, dim=-1)
            del selected_prefill_taps
        else:
            relay_features = extract_hidden_taps(
                target_prefill.hidden_states,
                relay_target_layer_ids,
            )
        conditioned_context = draft.hidden_norm(relay(relay_features))
    if release_capture_buffers:
        del target_prefill, relay_features
    torch.cuda.synchronize()
    time_to_first_token = time.perf_counter() - prefill_started

    start = num_input_tokens
    acceptance_lengths: list[int] = []
    proposal_lengths: list[int] = []
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
            # Match released DFlash: proposals are greedy even when target tokens sample.
            block_ids[:, 1:] = greedy_sample(draft_logits)

        with _profile_region(profile_recorder, "verification_full_target"):
            if selective_capture:
                target_verification, selected_verification_taps = forward_selected_taps(
                    native_target, block_ids, relay_target_layer_ids,
                    past_key_values=target_cache, use_cache=True,
                )
            else:
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
        proposal_lengths.append(int(block_ids.shape[1]) - 1)
        with _profile_region(profile_recorder, "relay"):
            if selective_capture:
                relay_features = torch.cat(selected_verification_taps, dim=-1)[:, :accepted, :]
                del selected_verification_taps
            else:
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
        proposal_lengths=proposal_lengths,
        accepted_draft_lengths=[x - 1 for x in acceptance_lengths],
        target_calls=target_calls,
        draft_calls=draft_calls,
    )


@torch.inference_mode()
def cross_family_relay_dflash_generate(
    draft: torch.nn.Module,
    *,
    relay: TargetFeatureRelay,
    relay_target_layer_ids: tuple[int, ...],
    native_target: torch.nn.Module,
    source_embedding: torch.nn.Module,
    source_lm_head: torch.nn.Module,
    source_tokenizer: Any,
    target_tokenizer: Any,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: list[int] | None,
    temperature: float,
    block_size: int | None = None,
    mask_token_id: int | None = None,
    return_stats: bool = False,
    profile_recorder: Any | None = None,
    source_to_target_intersection: dict[int, int] | None = None,
    target_to_source_intersection: dict[int, int] | None = None,
) -> Any:
    """S-A cross-family retargeting: verify the frozen proposer at string
    granularity through a target model that does not share its tokenizer.

    `input_ids` are in the target's own vocabulary. The frozen proposer only
    ever speaks its source vocabulary, so each cycle's speculative block is
    decoded to text and re-encoded in the target vocabulary before the
    target verifies it, and each cycle's newly verified text is decoded and
    re-encoded back into the source vocabulary to keep advancing the
    proposer. The target is still the sole commit authority: every emitted
    token is the target's own greedy choice, and agreement is only measured
    at a coarser, string-derived granularity than same-family speculative
    decoding. See
    docs/plans/2026-09-02-relayspec-source-free-adaptive-relay-plan.md
    section 3.5b (solution S-A).

    `source_to_target_intersection` and `target_to_source_intersection` are
    the optional S-B vocabulary-intersection maps (built once per tokenizer
    pair by `vocab_bridge.build_vocab_intersection`). When a proposed or
    committed chunk's ids are fully covered by the relevant map, the chunk
    is re-tokenized by direct id lookup instead of a decode/re-encode round
    trip; a partially covered chunk still falls back to the full S-A round
    trip. This never changes what gets verified or committed, only how
    often the (identical, by construction) re-tokenized result is reached.
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
    device = input_ids.device
    from transformers import DynamicCache

    target_cache = DynamicCache(config=native_target.config)
    draft_cache = DynamicCache()

    num_input_tokens = input_ids.shape[1]
    max_committed = num_input_tokens + max_new_tokens

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
    with _profile_region(profile_recorder, "prefill_relay"):
        relay_features = extract_hidden_taps(
            target_prefill.hidden_states,
            relay_target_layer_ids,
        )
        conditioned_context = draft.hidden_norm(relay(relay_features))
    # The draft never attends over the prompt directly, in the same-family
    # algorithm either: all prompt context reaches it through
    # `conditioned_context`, and its own cache only ever tracks generated
    # positions. So `draft_cache` starts empty here too, and the source-vocab
    # anchor for cycle one is the re-encoding of everything committed so far
    # (prompt plus the one token this prefill just generated), not the
    # prompt alone.
    target_committed_ids = torch.cat(
        [input_ids, greedy_sample(target_prefill.logits, temperature)], dim=1
    )
    committed_text_so_far = target_tokenizer.decode(
        target_committed_ids[0].tolist(), skip_special_tokens=False
    )
    source_committed_ids = encode_new_chunk(
        source_tokenizer, committed_text_so_far, device
    )
    if source_committed_ids.shape[1] == 0:
        raise ValueError("committed text re-encodes to zero source tokens")
    torch.cuda.synchronize()
    time_to_first_token = time.perf_counter() - prefill_started

    acceptance_lengths: list[int] = []
    target_calls = 0
    draft_calls = 0
    torch.cuda.synchronize()
    decode_started = time.perf_counter()
    while target_committed_ids.shape[1] < max_committed:
        with _profile_region(profile_recorder, "draft"):
            anchor_source_id = source_committed_ids[:, -1:]
            block_ids_source = torch.cat(
                [
                    anchor_source_id,
                    torch.full(
                        (1, block_size - 1),
                        mask_token_id,
                        dtype=torch.long,
                        device=device,
                    ),
                ],
                dim=1,
            )
            noise_embedding = source_embedding(block_ids_source)
            # The DFlash attention layer concatenates a key/value derived
            # from `conditioned_context` with one derived from
            # `noise_embedding`, then applies rotary embeddings to the whole
            # concatenation with no length adjustment. `position_ids` must
            # therefore span exactly `ctx_len + block_size` positions, and
            # after this forward the cache must be cropped back to keep only
            # the `ctx_len` (conditioned-context) portion: the block/noise
            # portion is always speculative and is discarded every cycle,
            # same as the same-family algorithm's own cache never retaining
            # a rejected proposal.
            pre_block_length = draft_cache.get_seq_length()
            ctx_len = conditioned_context.shape[1]
            draft_position_ids = torch.arange(
                pre_block_length,
                pre_block_length + ctx_len + block_size,
                device=device,
            ).unsqueeze(0)
            draft_hidden = _conditioned_dflash_forward(
                draft,
                conditioned_context=conditioned_context,
                noise_embedding=noise_embedding,
                position_ids=draft_position_ids,
                past_key_values=draft_cache,
            )
            draft_calls += 1
            draft_logits = source_lm_head(draft_hidden[:, 1 - block_size :, :])
            draft_cache.crop(pre_block_length + ctx_len)
            proposed_source_ids = greedy_sample(draft_logits, temperature)

        proposed_text = decode_new_chunk(source_tokenizer, proposed_source_ids[0])
        if source_to_target_intersection is not None:
            proposed_target_ids = bridge_encode_new_chunk(
                proposed_source_ids[0],
                proposed_text,
                target_tokenizer,
                device,
                source_to_target_intersection,
            )
        else:
            proposed_target_ids = encode_new_chunk(
                target_tokenizer, proposed_text, device
            )

        with _profile_region(profile_recorder, "verification_full_target"):
            anchor_target_id = target_committed_ids[:, -1:]
            target_block_ids = torch.cat([anchor_target_id, proposed_target_ids], dim=1)
            pre_verify_length = target_cache.get_seq_length()
            target_verification = native_target(
                target_block_ids,
                past_key_values=target_cache,
                use_cache=True,
                output_hidden_states=True,
            )
        target_calls += 1
        posterior = greedy_sample(target_verification.logits, temperature)
        if target_block_ids.shape[1] >= 2:
            accepted = accepted_block_length(target_block_ids, posterior)
        else:
            accepted = 1
        committed_this_cycle = committed_suffix(target_block_ids, posterior, accepted)
        target_cache.crop(pre_verify_length + accepted)
        target_committed_ids = torch.cat(
            [target_committed_ids, committed_this_cycle], dim=1
        )
        acceptance_lengths.append(accepted)

        with _profile_region(profile_recorder, "relay"):
            relay_features = extract_hidden_taps(
                target_verification.hidden_states,
                relay_target_layer_ids,
            )[:, :accepted, :]
            conditioned_context = draft.hidden_norm(relay(relay_features))

        # The draft's cache never actually holds token-embedding history (see
        # above): its only role here is to carry `conditioned_context`
        # forward across cycles, which already happened above. All that is
        # still needed is a fresh source-vocab anchor for the next cycle's
        # block, from whatever text was just committed.
        chunk_text = decode_new_chunk(target_tokenizer, committed_this_cycle[0])
        if target_to_source_intersection is not None:
            new_source_ids = bridge_encode_new_chunk(
                committed_this_cycle[0],
                chunk_text,
                source_tokenizer,
                device,
                target_to_source_intersection,
            )
        else:
            new_source_ids = encode_new_chunk(source_tokenizer, chunk_text, device)
        if new_source_ids.shape[1] > 0:
            source_committed_ids = torch.cat(
                [source_committed_ids, new_source_ids], dim=1
            )

        if stop_token_ids is not None:
            generated = target_committed_ids[:, num_input_tokens:]
            stops = torch.tensor(stop_token_ids, device=device)
            if torch.isin(generated, stops).any().item():
                break

    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_started
    output_ids = target_committed_ids[
        :, : min(max_committed, target_committed_ids.shape[1])
    ]
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
