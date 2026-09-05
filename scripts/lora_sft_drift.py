#!/usr/bin/env python3
"""Controlled SFT dose-response study (Phase A of the dynamic-relay-alignment
plan): LoRA fine-tune a target model for a fixed step budget, checkpointing
at several points, merging the adapter into standalone weights at each one.

Each merged checkpoint is then benchmarkable with the existing, unmodified
benchmark_relay.py, exactly like E3 Stage A's public-descendant experiment
(docs/research/relayspec-decision-register.md, "E3 checkpoint verified and
found"), just substituting a locally-produced checkpoint directory for a
HuggingFace-hosted one.

Data: HuggingFaceH4/ultrachat_200k (general instruction-following), chosen
specifically because it does not overlap with configs/train_math_4096.json
(MATH-lighteval, used for all relay fitting in this project) or with the
MATH-500/GSM8K/HumanEval/MBPP evaluation sets, so fine-tuning here cannot be
confounded with the interface-regression data the relay itself was fit on.

Loss is standard full-sequence next-token cross-entropy over the formatted
multi-turn conversation (no response-only masking): a simpler convention
than response-masked SFT, used here because this is a controlled-drift
probe, not a quality-optimized fine-tune, and stated in the paper as
exactly that.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_ultrachat_shard(
    cache_dir: str, revision: str, num_examples: int
) -> list[dict[str, Any]]:
    path = hf_hub_download(
        repo_id="HuggingFaceH4/ultrachat_200k",
        repo_type="dataset",
        revision=revision,
        filename="data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
        cache_dir=cache_dir,
    )
    table = pq.read_table(path, columns=["messages"])
    rows = table.to_pylist()[:num_examples]
    return rows


def encode_conversation(
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_length: int,
    device: torch.device,
) -> torch.LongTensor:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    return input_ids[:, :max_length].to(device)


def save_merged_checkpoint(peft_model: Any, tokenizer: Any, output_dir: Path) -> None:
    """Save a standalone, plain-format checkpoint without disturbing training.

    `merge_and_unload` permanently removes the LoRA adapter layers, which
    would make it impossible to resume training past this checkpoint.
    `merge_adapter`/`unmerge_adapter` fold the LoRA delta into the base
    layers' weights and back out again, reversibly, so training continues
    exactly as if nothing happened once we unmerge.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    peft_model.merge_adapter()
    try:
        peft_model.base_model.model.save_pretrained(output_dir, safe_serialization=True)
    finally:
        peft_model.unmerge_adapter()
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    training = json.loads(args.config.read_text(encoding="utf-8"))

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise RuntimeError("lora sft drift study requires exactly four GPUs")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", device_id=torch.device(f"cuda:{local_rank}"), timeout=timedelta(hours=4)
    )
    device = torch.device(f"cuda:{local_rank}")
    cache_dir = Path(os.environ["TRANSFORMERS_CACHE"])
    output_root = Path(os.environ["RELAYSPEC_OUTPUT"])
    # Merged full-model checkpoints are ~16 GiB each (Qwen3-8B, bf16) and
    # must never go under $RELAYSPEC_OUTPUT: that lives on the home
    # filesystem's small, shared quota (already exhausted once this
    # session). Only small metrics/summary files go there; checkpoints go
    # to the node-local /scratch cache directory alongside the model cache.
    checkpoint_root = (
        Path(os.environ["RELAYSPEC_CACHE_DIR"])
        / "relayspec"
        / "lora_sft_checkpoints"
        / training.get("run_name", "default")
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        training["base_model_id"],
        revision=training["base_model_revision"],
        cache_dir=cache_dir,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        training["base_model_id"],
        revision=training["base_model_revision"],
        cache_dir=cache_dir,
        local_files_only=True,
    )

    lora_config = LoraConfig(
        r=int(training.get("lora_rank", 16)),
        lora_alpha=int(training.get("lora_alpha", 32)),
        target_modules=list(
            training.get(
                "lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
            )
        ),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.train()
    trainable_parameters = [p for p in peft_model.parameters() if p.requires_grad]
    # DDP synchronizes gradients across the 4 ranks so every rank ends the
    # step with the identical LoRA update, matching train_relay.py's own
    # DDP-wrapped-trainable-module pattern. peft freezes every base-model
    # parameter by default, so only the LoRA adapters are ever synchronized.
    model = torch.nn.parallel.DistributedDataParallel(
        peft_model,
        device_ids=[local_rank],
        find_unused_parameters=False,
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=0.0,
    )

    total_steps = int(training["total_steps"])
    checkpoint_steps = set(int(s) for s in training["checkpoint_steps"])
    max_length = int(training["max_length"])
    dataset_revision = training["dataset_revision"]

    dataset_cache_dir = os.environ.get(
        "HF_DATASETS_CACHE", str(cache_dir / "hf-datasets")
    )
    rows = load_ultrachat_shard(
        dataset_cache_dir, dataset_revision, total_steps * world_size * 2
    )
    local_rows = rows[rank::world_size]

    metrics_path = output_root / f"lora-sft-rank{rank}.jsonl"
    with metrics_path.open("x", encoding="utf-8") as metrics:
        row_index = 0
        for step in range(1, total_steps + 1):
            row = local_rows[row_index % len(local_rows)]
            row_index += 1
            input_ids = encode_conversation(
                tokenizer, row["messages"], max_length, device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(input_ids, labels=input_ids)
                loss = output.loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()
            metrics.write(
                json.dumps(
                    {
                        "step": step,
                        "rank": rank,
                        "loss": float(loss.item()),
                        "grad_norm": float(grad_norm),
                        "tokens": int(input_ids.shape[1]),
                    }
                )
                + "\n"
            )
            metrics.flush()

            if step in checkpoint_steps:
                dist.barrier()
                if rank == 0:
                    checkpoint_dir = checkpoint_root / f"checkpoint-step-{step}"
                    save_merged_checkpoint(model.module, tokenizer, checkpoint_dir)
                    print(
                        json.dumps(
                            {
                                "event": "checkpoint_saved",
                                "step": step,
                                "path": str(checkpoint_dir),
                            }
                        )
                    )
                dist.barrier()

    dist.barrier()
    if rank == 0:
        summary = {
            "total_steps": total_steps,
            "checkpoint_steps": sorted(checkpoint_steps),
            "checkpoint_root": str(checkpoint_root),
            "checkpoint_paths": {
                str(step): str(checkpoint_root / f"checkpoint-step-{step}")
                for step in sorted(checkpoint_steps)
            },
        }
        (output_root / "lora-sft-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    dist.barrier()


if __name__ == "__main__":
    main()
