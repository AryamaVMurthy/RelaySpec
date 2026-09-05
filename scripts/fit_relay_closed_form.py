#!/usr/bin/env python3
"""Closed-form ridge regression baseline for the relay (rigor-pass ablation).

The relay is a single linear map. Instead of fitting it by gradient descent
(every other result in this paper), solve the ridge-regularized normal
equations directly: R = (Z^T Z + lambda I)^-1 Z^T C, where Z is the matrix
of normalized target taps across every fitting position and C is the
matching source-derived teacher tensor. This answers a direct question a
reviewer will ask: is gradient descent even necessary for a linear
hypothesis class, or does the closed-form least-squares solution work as
well? Runs the same DDP world as train_relay.py so the normal-equation
sums (which are just sums over examples) reduce across ranks with one
all_reduce, but there is only one linear-algebra solve at the end, not many
optimizer steps.

Saves a checkpoint in the exact same format train_relay.py writes, so it
can be benchmarked with the unmodified benchmark_relay.py.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from relayspec.dflash import import_official_dflash
from relayspec.proposers import project_source_interface
from relayspec.relay import extract_hidden_taps


def namespace(payload: dict[str, Any]) -> Any:
    from types import SimpleNamespace

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return SimpleNamespace(**{k: convert(v) for k, v in value.items()})
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value

    return convert(payload)


def encode_example(
    tokenizer: Any, row: dict[str, Any], max_length: int, device: torch.device
) -> torch.LongTensor:
    encoded = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": row["problem"]},
            {"role": "assistant", "content": row["solution"]},
        ],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    return input_ids[:, :max_length].to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training = payload.pop("relay_training")
    config = namespace(payload)
    source_config = config.source_trunk

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise RuntimeError("closed-form fit requires exactly four GPUs")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", device_id=torch.device(f"cuda:{local_rank}"), timeout=timedelta(hours=2)
    )
    device = torch.device(f"cuda:{local_rank}")
    cache_dir = Path(os.environ["TRANSFORMERS_CACHE"])

    draft_class, _ = import_official_dflash(
        os.environ["DFLASH_SOURCE"], config.proposer.source_commit
    )
    source = (
        AutoModelForCausalLM.from_pretrained(
            source_config.model.id,
            revision=source_config.model.revision,
            cache_dir=cache_dir,
            attn_implementation="sdpa",
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    target = (
        AutoModelForCausalLM.from_pretrained(
            config.target.id,
            revision=config.target.revision,
            cache_dir=cache_dir,
            attn_implementation="sdpa",
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    draft = (
        draft_class.from_pretrained(
            config.proposer.id,
            revision=config.proposer.revision,
            cache_dir=cache_dir,
            attn_implementation="sdpa",
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    for m in (source, target, draft):
        m.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(
        config.target.id,
        revision=config.target.revision,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    target_layer_ids = tuple(int(v) for v in training["target_layer_ids"])
    fit_examples = int(training["fit_examples"])
    max_length = int(training["max_length"])
    ridge_lambda = float(training.get("ridge_lambda", 1.0))

    manifest = json.loads(Path(training["manifest_path"]).read_text(encoding="utf-8"))
    rows = manifest["records"][:fit_examples]
    local_rows = rows[rank::world_size]

    input_width = target.config.hidden_size * len(target_layer_ids)
    draft_hidden_size = draft.config.hidden_size
    ztz = torch.zeros((input_width, input_width), dtype=torch.float64, device=device)
    ztc = torch.zeros(
        (input_width, draft_hidden_size), dtype=torch.float64, device=device
    )
    input_norm = torch.nn.RMSNorm(
        input_width, eps=target.config.rms_norm_eps, elementwise_affine=False
    ).to(device)

    total_positions = 0
    started = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    started.record()
    with torch.no_grad():
        for row in local_rows:
            input_ids = encode_example(tokenizer, row, max_length, device)
            source_output = source(
                input_ids, use_cache=False, output_hidden_states=True, logits_to_keep=1
            )
            source_features = extract_hidden_taps(
                source_output.hidden_states,
                tuple(int(v) for v in draft.target_layer_ids),
            )
            teacher_context = project_source_interface(
                draft, source_features, family="dflash"
            )
            target_output = target(
                input_ids, use_cache=False, output_hidden_states=True, logits_to_keep=1
            )
            target_features = extract_hidden_taps(
                target_output.hidden_states, target_layer_ids
            )
            target_features = target_features[:, : input_ids.shape[1], :]

            z = input_norm(target_features).squeeze(0).double()  # [T, input_width]
            c = teacher_context.squeeze(0).double()  # [T, draft_hidden]
            # relative_interface_mse (the loss gradient descent actually
            # minimizes, src/relayspec/losses.py) weights each position's
            # squared error by 1/||c_t||^2. Since R is linear,
            # ||R z_t - c_t||^2 / ||c_t||^2 == ||R (z_t/||c_t||) - c_t/||c_t||||^2,
            # so pre-scaling each row by 1/||c_t|| before accumulating the
            # normal equations makes this closed-form solve the identical
            # weighted least-squares problem SGD optimizes. Without this
            # scaling the two fitting procedures solve different objectives
            # (this one plain, unweighted ridge regression), which
            # confounds any closed-form-vs-gradient-descent comparison.
            position_energy = (
                c.square()
                .sum(dim=-1, keepdim=True)
                .clamp_min(torch.finfo(torch.float64).tiny)
            )
            weight = position_energy.rsqrt()
            z_weighted = z * weight
            c_weighted = c * weight
            ztz += z_weighted.T @ z_weighted
            ztc += z_weighted.T @ c_weighted
            total_positions += z.shape[0]

    ended.record()
    torch.cuda.synchronize()

    dist.all_reduce(ztz, op=dist.ReduceOp.SUM)
    dist.all_reduce(ztc, op=dist.ReduceOp.SUM)
    total_positions_tensor = torch.tensor(
        [total_positions], device=device, dtype=torch.float64
    )
    dist.all_reduce(total_positions_tensor, op=dist.ReduceOp.SUM)
    total_positions = int(total_positions_tensor.item())

    if rank == 0:
        regularized = ztz + ridge_lambda * torch.eye(
            input_width, dtype=torch.float64, device=device
        )
        solve_started = torch.cuda.Event(enable_timing=True)
        solve_ended = torch.cuda.Event(enable_timing=True)
        solve_started.record()
        weight = torch.linalg.solve(regularized, ztc)  # [input_width, draft_hidden]
        solve_ended.record()
        torch.cuda.synchronize()

        # Matches train_relay.py's own checkpoint convention exactly
        # ($RELAYSPEC_CACHE_DIR/relayspec/checkpoints/<run_name>/relay.pt,
        # on node-local /scratch), not $RELAYSPEC_OUTPUT (the small,
        # shared-quota home filesystem every benchmark config's
        # relay_probe.checkpoint_path already expects this checkpoint at).
        checkpoint_dir = (
            Path(os.environ["RELAYSPEC_CACHE_DIR"])
            / "relayspec"
            / "checkpoints"
            / config.run_name
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "relay": {"projection.weight": weight.T.to(torch.float32).cpu()},
            "target_layer_ids": list(target_layer_ids),
            "relay_architecture": "normalized_linear",
        }
        torch.save(checkpoint, checkpoint_dir / "relay.pt")
        output_dir = Path(os.environ["RELAYSPEC_OUTPUT"])
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "total_positions": total_positions,
            "fit_examples": fit_examples,
            "ridge_lambda": ridge_lambda,
            "checkpoint_path": str(checkpoint_dir / "relay.pt"),
            "collect_seconds": started.elapsed_time(ended) / 1000.0,
            "solve_seconds": solve_started.elapsed_time(solve_ended) / 1000.0,
            "input_width": input_width,
        }
        (output_dir / "closed-form-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps({"event": "closed_form_fit", **summary}))
    dist.barrier()


if __name__ == "__main__":
    main()
