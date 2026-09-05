#!/usr/bin/env python3
"""Step 1 of docs/plans/2026-09-03-relayspec-cross-family-fix-plan.md.

Runs native_ar and cross_family_relay_dflash_generate on the same prompt,
single GPU, and finds the first position where the two committed token
streams diverge. At that position, compares the target's own logit gap
between the two candidate tokens: a near-tied gap (under 1e-2 in
log-probability) is consistent with the already-documented block-versus-
single-position kernel effect (docs/.../relayspec_iclr2027.tex,
"Greedy correctness and finite precision"). A large, clearly-not-tied gap
means the two code paths fed the model different state, a real bug rather
than numerical noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from relayspec.dflash import import_official_dflash
from relayspec.generation import (
    cross_family_relay_dflash_generate,
    native_autoregressive_generate,
)
from relayspec.relay import TargetFeatureRelay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--target-revision", required=True)
    parser.add_argument("--proposer-id", required=True)
    parser.add_argument("--proposer-revision", required=True)
    parser.add_argument("--proposer-source-commit", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--relay-checkpoint", required=True)
    parser.add_argument("--dflash-source", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    draft_class, _ = import_official_dflash(
        args.dflash_source, args.proposer_source_commit
    )

    target = (
        AutoModelForCausalLM.from_pretrained(
            args.target_id,
            revision=args.target_revision,
            cache_dir=args.cache_dir,
            attn_implementation="sdpa",
            dtype=dtype,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    source = (
        AutoModelForCausalLM.from_pretrained(
            args.source_id,
            revision=args.source_revision,
            cache_dir=args.cache_dir,
            attn_implementation="sdpa",
            dtype=dtype,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    draft = (
        draft_class.from_pretrained(
            args.proposer_id,
            revision=args.proposer_revision,
            cache_dir=args.cache_dir,
            attn_implementation="sdpa",
            dtype=dtype,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    target_tokenizer = AutoTokenizer.from_pretrained(
        args.target_id,
        revision=args.target_revision,
        cache_dir=args.cache_dir,
        local_files_only=True,
    )
    source_tokenizer = AutoTokenizer.from_pretrained(
        args.source_id,
        revision=args.source_revision,
        cache_dir=args.cache_dir,
        local_files_only=True,
    )
    checkpoint = torch.load(
        args.relay_checkpoint, map_location="cpu", weights_only=True
    )
    target_layer_ids = tuple(int(v) for v in checkpoint["target_layer_ids"])
    relay = TargetFeatureRelay(
        target_hidden_size=target.config.hidden_size,
        num_taps=len(target_layer_ids),
        draft_hidden_size=draft.config.hidden_size,
        eps=target.config.rms_norm_eps,
        normalize_input=checkpoint.get("relay_architecture", "normalized_linear")
        == "normalized_linear",
    )
    relay.load_state_dict(checkpoint["relay"], strict=True)
    relay = relay.to(device=device, dtype=dtype).eval()

    encoded_prompt = target_tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    input_ids = (
        encoded_prompt["input_ids"]
        if hasattr(encoded_prompt, "keys")
        else encoded_prompt
    ).to(device)

    with torch.inference_mode():
        ar_out = native_autoregressive_generate(
            target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=[target_tokenizer.eos_token_id],
            temperature=0.0,
            return_stats=True,
        )
        cf_out = cross_family_relay_dflash_generate(
            draft,
            relay=relay,
            relay_target_layer_ids=target_layer_ids,
            native_target=target,
            source_embedding=source.model.embed_tokens,
            source_lm_head=source.lm_head,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=[target_tokenizer.eos_token_id],
            temperature=0.0,
            block_size=args.block_size,
            return_stats=True,
        )

    ar_ids = ar_out.output_ids[0].tolist()
    cf_ids = cf_out.output_ids[0].tolist()
    num_input = input_ids.shape[1]
    ar_new = ar_ids[num_input:]
    cf_new = cf_ids[num_input:]

    first_divergence = None
    for i in range(min(len(ar_new), len(cf_new))):
        if ar_new[i] != cf_new[i]:
            first_divergence = i
            break

    result = {
        "prompt": args.prompt,
        "ar_text": target_tokenizer.decode(ar_new, skip_special_tokens=True),
        "cf_text": target_tokenizer.decode(cf_new, skip_special_tokens=True),
        "ar_len": len(ar_new),
        "cf_len": len(cf_new),
        "first_divergence_index": first_divergence,
        "match": ar_new == cf_new,
    }

    if first_divergence is not None:
        # Recompute the target's own logits at the diverging position under
        # both committed-prefix histories, to check whether the two
        # candidate tokens were near-tied (kernel noise) or not (real bug).
        prefix_ar = torch.tensor(
            [ar_ids[: num_input + first_divergence]], device=device
        )
        prefix_cf = torch.tensor(
            [cf_ids[: num_input + first_divergence]], device=device
        )
        prefixes_match = prefix_ar.shape == prefix_cf.shape and torch.equal(
            prefix_ar, prefix_cf
        )
        result["prefixes_match_before_divergence"] = bool(prefixes_match)
        with torch.inference_mode():
            logits_ar = target(prefix_ar, use_cache=False).logits[0, -1, :].float()
        log_probs = F.log_softmax(logits_ar, dim=-1)
        ar_token = ar_new[first_divergence]
        cf_token = cf_new[first_divergence]
        result["ar_token"] = target_tokenizer.decode([ar_token])
        result["cf_token"] = target_tokenizer.decode([cf_token])
        result["log_prob_gap"] = float(
            log_probs[ar_token].item() - log_probs[cf_token].item()
        )
        result["diagnosis"] = (
            "near-tied logit gap, consistent with block-vs-single-position kernel noise"
            if abs(result["log_prob_gap"]) < 1e-2
            else "large logit gap: the two paths fed the model different state, a real bug"
        )

    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "diagnosis", **result}, default=str))


if __name__ == "__main__":
    main()
