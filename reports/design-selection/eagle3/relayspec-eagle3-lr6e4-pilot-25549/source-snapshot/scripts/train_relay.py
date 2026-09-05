from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from itertools import cycle, islice
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer

from relayspec.dflash import import_official_dflash
from relayspec.eagle3 import import_official_eagle3
from relayspec.generation import _conditioned_dflash_forward
from relayspec.losses import interface_alignment_loss, proposal_kl_loss
from relayspec.proposers import project_source_interface
from relayspec.relay import TargetFeatureRelay, extract_hidden_taps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [namespace(item) for item in value]
    return value


def encode_example(
    tokenizer: Any,
    row: dict[str, Any],
    max_length: int,
    device: torch.device,
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
    args = parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training = payload.pop("relay_training")
    config = namespace(payload)
    source_config = config.source_trunk
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4 or config.resources.gpu_count != 4:
        raise RuntimeError("target relay training requires exactly four GPUs")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
        timeout=timedelta(hours=6),
    )
    device = torch.device(f"cuda:{local_rank}")
    cache_dir = Path(os.environ["TRANSFORMERS_CACHE"])
    torch.manual_seed(config.seed + rank)
    torch.cuda.manual_seed_all(config.seed + rank)

    proposer_family = str(config.proposer.family)
    if proposer_family == "dflash":
        draft_class, _ = import_official_dflash(
            os.environ["DFLASH_SOURCE"], config.proposer.source_commit
        )
    elif proposer_family == "eagle3":
        draft_class = import_official_eagle3(
            os.environ["DEEPSPEC_SOURCE"], config.proposer.source_commit
        ).model_class
    else:
        raise ValueError(f"unsupported proposer family: {proposer_family}")
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
    for frozen_model in (source, target, draft):
        frozen_model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(
        config.target.id,
        revision=config.target.revision,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    target_layer_ids = tuple(int(value) for value in training["target_layer_ids"])
    if len(target_layer_ids) != len(draft.target_layer_ids):
        raise ValueError("target relay tap count must match the trained proposer")
    relay = TargetFeatureRelay(
        target_hidden_size=target.config.hidden_size,
        num_taps=len(target_layer_ids),
        draft_hidden_size=draft.config.hidden_size,
        eps=target.config.rms_norm_eps,
    ).to(device)
    initial_checkpoint = os.environ.get(
        "RELAYSPEC_INITIAL_CHECKPOINT",
        training.get("initial_checkpoint_path"),
    )
    if initial_checkpoint is not None:
        initial = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
        if (
            tuple(int(value) for value in initial["target_layer_ids"])
            != target_layer_ids
        ):
            raise ValueError("initial relay target taps do not match current training")
        relay.load_state_dict(initial["relay"], strict=True)
    distributed = DistributedDataParallel(relay, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        distributed.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    records = json.loads(Path(training["manifest_path"]).read_text(encoding="utf-8"))[
        "records"
    ]
    local_rows = records[rank::world_size]
    steps = int(training["steps"])
    expected_fit_examples = int(training.get("fit_examples", 0))
    if expected_fit_examples and expected_fit_examples != steps * world_size:
        raise ValueError(
            "fit_examples must equal steps times world_size for one-example-per-rank DDP"
        )
    proposal_weight = float(training.get("proposal_kl_weight", 0.0))
    feature_objective = str(training["feature_objective"])
    historical_cosine_weight = training.get("historical_cosine_weight")
    proposal_temperature = float(training.get("proposal_temperature", 1.0))
    proposal_block_size = int(
        training.get("proposal_block_size", getattr(draft, "block_size", 2))
    )
    if proposal_weight < 0:
        raise ValueError("proposal KL weight must be non-negative")
    if proposal_weight > 0 and proposer_family != "dflash":
        raise ValueError("proposal KL refinement is currently DFlash-specific")
    if proposal_weight > 0 and proposal_block_size < 2:
        raise ValueError("proposal-aligned training requires block size at least two")
    output_dir = Path(os.environ["RELAYSPEC_OUTPUT"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"relay-train-rank{rank}.jsonl"
    started = time.perf_counter()
    with metrics_path.open("x", encoding="utf-8") as metrics:
        iterator = islice(cycle(local_rows), steps)
        for step, row in enumerate(iterator, start=1):
            input_ids = encode_example(
                tokenizer,
                row,
                int(training["max_length"]),
                device,
            )
            feature_ids = input_ids
            if proposal_weight > 0:
                if input_ids.shape[1] <= proposal_block_size:
                    raise ValueError("proposal example is shorter than its draft block")
                feature_ids = input_ids[:, :-proposal_block_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                source_output = source(
                    feature_ids,
                    use_cache=False,
                    output_hidden_states=True,
                    logits_to_keep=1,
                )
                source_features = extract_hidden_taps(
                    source_output.hidden_states,
                    tuple(int(value) for value in draft.target_layer_ids),
                )
                teacher_context = project_source_interface(
                    draft,
                    source_features,
                    family=proposer_family,
                ).detach()
                target_output = target(
                    feature_ids,
                    use_cache=False,
                    output_hidden_states=True,
                    logits_to_keep=1,
                )
                target_features = extract_hidden_taps(
                    target_output.hidden_states,
                    target_layer_ids,
                ).detach()
                teacher_logits = None
                noise_embedding = None
                position_ids = None
                if proposal_weight > 0:
                    block_ids = torch.full(
                        (1, proposal_block_size),
                        int(draft.mask_token_id),
                        dtype=torch.long,
                        device=device,
                    )
                    block_ids[:, 0] = input_ids[:, feature_ids.shape[1]]
                    noise_embedding = source.model.embed_tokens(block_ids)
                    position_ids = torch.arange(
                        feature_ids.shape[1] + proposal_block_size,
                        device=device,
                    ).unsqueeze(0)
                    teacher_hidden = draft(
                        target_hidden=source_features,
                        noise_embedding=noise_embedding,
                        position_ids=position_ids,
                        past_key_values=None,
                        use_cache=False,
                        is_causal=False,
                    )
                    teacher_logits = source.lm_head(
                        teacher_hidden[:, 1 - proposal_block_size :, :]
                    ).detach()
                del source_output, source_features, target_output
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predicted_context = distributed(target_features)
                if proposer_family == "dflash":
                    predicted_context = draft.hidden_norm(predicted_context)
            mse = F.mse_loss(predicted_context.float(), teacher_context.float())
            cosine = (
                1.0
                - F.cosine_similarity(
                    predicted_context.float(),
                    teacher_context.float(),
                    dim=-1,
                ).mean()
            )
            proposal_kl = torch.zeros((), device=device)
            if proposal_weight > 0:
                assert noise_embedding is not None
                assert position_ids is not None
                assert teacher_logits is not None
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predicted_hidden = _conditioned_dflash_forward(
                        draft,
                        conditioned_context=predicted_context,
                        noise_embedding=noise_embedding,
                        position_ids=position_ids,
                        past_key_values=None,
                    )
                    predicted_logits = source.lm_head(
                        predicted_hidden[:, 1 - proposal_block_size :, :]
                    )
                proposal_kl = proposal_kl_loss(
                    predicted_logits,
                    teacher_logits,
                    temperature=proposal_temperature,
                )
            feature_loss = interface_alignment_loss(
                predicted_context,
                teacher_context,
                objective=feature_objective,
                historical_cosine_weight=historical_cosine_weight,
            )
            loss = feature_loss + proposal_weight * proposal_kl
            loss.backward()
            gradient_clip = training.get("gradient_clip")
            if gradient_clip is None:
                gradient_norm = torch.linalg.vector_norm(
                    torch.stack(
                        [
                            parameter.grad.detach().float().norm()
                            for parameter in distributed.parameters()
                            if parameter.grad is not None
                        ]
                    )
                )
            else:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    distributed.parameters(),
                    float(gradient_clip),
                )
            optimizer.step()
            record = {
                "step": step,
                "rank": rank,
                "tokens": int(input_ids.shape[1]),
                "mse": float(mse.detach().item()),
                "cosine_distance": float(cosine.detach().item()),
                "proposal_kl": float(proposal_kl.detach().item()),
                "proposal_kl_weight": proposal_weight,
                "feature_objective": feature_objective,
                "feature_loss": float(feature_loss.detach().item()),
                "loss": float(loss.detach().item()),
                "gradient_norm": float(gradient_norm.detach().item()),
                "elapsed_seconds": time.perf_counter() - started,
                "initialized_from": initial_checkpoint,
                "proposer_family": proposer_family,
            }
            metrics.write(json.dumps(record, sort_keys=True) + "\n")
            metrics.flush()
            if rank == 0 and (step == 1 or step % 8 == 0):
                print(
                    json.dumps({"event": "relay_train", **record}, sort_keys=True),
                    flush=True,
                )
            del (
                target_features,
                teacher_context,
                predicted_context,
                feature_loss,
                loss,
                mse,
                cosine,
            )
            if proposal_weight > 0:
                del predicted_hidden, predicted_logits, teacher_logits

    dist.barrier()
    if rank == 0:
        checkpoint_dir = (
            Path(os.environ["RELAYSPEC_CACHE_DIR"])
            / "relayspec"
            / "checkpoints"
            / config.run_name
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "relay.pt"
        torch.save(
            {
                "relay": distributed.module.state_dict(),
                "target_layer_ids": target_layer_ids,
                "source_layer_ids": tuple(
                    int(value) for value in draft.target_layer_ids
                ),
                "steps": steps,
                "feature_objective": feature_objective,
                "weight_decay": float(training["weight_decay"]),
                "gradient_clip": training.get("gradient_clip"),
                "initialized_from": initial_checkpoint,
                "proposer_family": proposer_family,
            },
            checkpoint_path,
        )
        summary = {
            "status": "complete",
            "checkpoint": str(checkpoint_path),
            "steps": steps,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(local_rank),
            "initialized_from": initial_checkpoint,
        }
        (output_dir / "relay-training-summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps({"event": "relay_training_summary", **summary}, sort_keys=True)
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
