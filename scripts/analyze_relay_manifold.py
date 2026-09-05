#!/usr/bin/env python3
"""E4-P0: free representation-alignment probe between two targets.

Two modes.

Relay mode (original, E4-P0): loads two already-trained relay checkpoints
for different targets (e.g. the published Qwen3-8B and Qwen3-14B DFlash
relays), runs both targets over the same held-out text, and compares the
resulting c-hat trajectories in the shared 2,560-dimensional proposer
interface. No training, no proposer, no source model. See
docs/plans/2026-09-02-relayspec-source-free-adaptive-relay-plan.md
section 3.5, P0.

Raw mode (--raw-hidden-states, added for the policy-drift study): skips
relay loading entirely and compares the two targets' own raw tap features
directly, so it can compare a base model against a fine-tuned or
RL-drifted checkpoint of that same model with no relay needed at all. Set
`--target-a-tap-layers`/`--target-b-tap-layers` instead of
`--checkpoint-a`/`--checkpoint-b` in this mode. See
docs/plans/2026-09-02-relayspec-source-free-adaptive-relay-plan.md and
the dynamic-relay-alignment plan for how this drift metric is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from relayspec.relay import TargetFeatureRelay, extract_hidden_taps


def load_relay(
    checkpoint_path: str, device: torch.device
) -> tuple[TargetFeatureRelay, tuple[int, ...], dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    target_layer_ids = tuple(int(v) for v in payload["target_layer_ids"])
    state = payload["relay"]
    # DDP-wrapped checkpoints store parameters under a "module." prefix.
    state = {key.removeprefix("module."): value for key, value in state.items()}
    target_hidden_size = state["projection.weight"].shape[1] // len(target_layer_ids)
    draft_hidden_size = state["projection.weight"].shape[0]
    relay = TargetFeatureRelay(
        target_hidden_size=target_hidden_size,
        num_taps=len(target_layer_ids),
        draft_hidden_size=draft_hidden_size,
        eps=1e-6,
        normalize_input=payload.get("relay_architecture", "normalized_linear")
        == "normalized_linear",
    ).to(device)
    relay.load_state_dict(state, strict=True)
    relay = relay.to(device=device, dtype=torch.bfloat16)
    relay.eval()
    return relay, target_layer_ids, payload


def cka_linear(a: torch.Tensor, b: torch.Tensor) -> float:
    """Linear centered kernel alignment between two [N, D] feature matrices."""
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    hsic = (a.T @ b).norm() ** 2
    normalizer = (a.T @ a).norm() * (b.T @ b).norm()
    return float((hsic / normalizer.clamp_min(1e-12)).item())


def principal_subspace_overlap(
    a: torch.Tensor, b: torch.Tensor, rank: int = 32
) -> float:
    """Mean squared cosine principal angle between top-`rank` subspaces."""
    ua, _, _ = torch.linalg.svd(a - a.mean(dim=0, keepdim=True), full_matrices=False)
    ub, _, _ = torch.linalg.svd(b - b.mean(dim=0, keepdim=True), full_matrices=False)
    ua = ua[:, :rank]
    ub = ub[:, :rank]
    singular_values = torch.linalg.svdvals(ua.T @ ub)
    return float((singular_values**2).mean().item())


def _parse_tap_layers(value: str) -> tuple[int, ...]:
    return tuple(int(v) for v in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-hidden-states", action="store_true")
    parser.add_argument("--checkpoint-a")
    parser.add_argument("--checkpoint-b")
    parser.add_argument("--target-a-tap-layers", type=_parse_tap_layers)
    parser.add_argument("--target-b-tap-layers", type=_parse_tap_layers)
    parser.add_argument("--target-a-id", required=True)
    parser.add_argument("--target-a-revision", required=True)
    parser.add_argument("--target-b-id", required=True)
    parser.add_argument("--target-b-revision", required=True)
    parser.add_argument("--text-manifest", required=True)
    parser.add_argument("--num-examples", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.raw_hidden_states:
        if args.target_a_tap_layers is None or args.target_b_tap_layers is None:
            raise ValueError(
                "--raw-hidden-states requires --target-a-tap-layers and "
                "--target-b-tap-layers"
            )
    elif args.checkpoint_a is None or args.checkpoint_b is None:
        raise ValueError("relay mode requires --checkpoint-a and --checkpoint-b")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    relay_a = relay_b = None
    if args.raw_hidden_states:
        taps_a = args.target_a_tap_layers
        taps_b = args.target_b_tap_layers
    else:
        relay_a, taps_a, _ = load_relay(args.checkpoint_a, device)
        relay_b, taps_b, _ = load_relay(args.checkpoint_b, device)

    target_a = (
        AutoModelForCausalLM.from_pretrained(
            args.target_a_id,
            revision=args.target_a_revision,
            cache_dir=args.cache_dir,
            attn_implementation="sdpa",
            dtype=dtype,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    target_b = (
        AutoModelForCausalLM.from_pretrained(
            args.target_b_id,
            revision=args.target_b_revision,
            cache_dir=args.cache_dir,
            attn_implementation="sdpa",
            dtype=dtype,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    tokenizer_a = AutoTokenizer.from_pretrained(
        args.target_a_id,
        revision=args.target_a_revision,
        cache_dir=args.cache_dir,
        local_files_only=True,
    )
    tokenizer_b = AutoTokenizer.from_pretrained(
        args.target_b_id,
        revision=args.target_b_revision,
        cache_dir=args.cache_dir,
        local_files_only=True,
    )

    records = json.loads(Path(args.text_manifest).read_text(encoding="utf-8"))[
        "records"
    ]
    records = records[: args.num_examples]

    c_hat_a_rows: list[torch.Tensor] = []
    c_hat_b_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for row in records:
            for tokenizer, target, relay, taps, sink in (
                (tokenizer_a, target_a, relay_a, taps_a, c_hat_a_rows),
                (tokenizer_b, target_b, relay_b, taps_b, c_hat_b_rows),
            ):
                encoded = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": row["answer"]},
                    ],
                    tokenize=True,
                    add_generation_prompt=False,
                    enable_thinking=False,
                    return_tensors="pt",
                )
                input_ids = (
                    encoded["input_ids"] if hasattr(encoded, "keys") else encoded
                )[:, : args.max_length].to(device)
                output = target(
                    input_ids,
                    use_cache=False,
                    output_hidden_states=True,
                    logits_to_keep=1,
                )
                features = extract_hidden_taps(output.hidden_states, taps)
                c_hat = (
                    features[0].float().cpu()
                    if relay is None
                    else relay(features)[0].float().cpu()
                )
                sink.append(c_hat)

    c_hat_a = torch.cat(c_hat_a_rows, dim=0)
    c_hat_b = torch.cat(c_hat_b_rows, dim=0)
    length = min(c_hat_a.shape[0], c_hat_b.shape[0])
    c_hat_a = c_hat_a[:length]
    c_hat_b = c_hat_b[:length]

    cosine = torch.nn.functional.cosine_similarity(c_hat_a, c_hat_b, dim=-1)
    result = {
        "mode": "raw_hidden_states" if args.raw_hidden_states else "relay",
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "target_a": args.target_a_id,
        "target_b": args.target_b_id,
        "num_tokens_compared": length,
        "elementwise_cosine_mean": float(cosine.mean().item()),
        "elementwise_cosine_std": float(cosine.std().item()),
        "cka_linear": cka_linear(c_hat_a, c_hat_b),
        "principal_subspace_overlap_rank32": principal_subspace_overlap(
            c_hat_a, c_hat_b, rank=min(32, length)
        ),
    }
    Path(args.output).write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "manifold_probe", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
