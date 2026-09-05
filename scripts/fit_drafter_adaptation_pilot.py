"""Four bounded CUDA checks for connector-only and inherited-drafter updates."""

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM

from relayspec.dflash import import_official_dflash
from relayspec.drafter_adaptation import (
    apply_merged_lora,
    export_merged_lora,
    frozen_base_digest,
)
from relayspec.generation import _conditioned_dflash_forward
from relayspec.lora import inject_lora, merge_lora_into_base
from relayspec.losses import greedy_agreement_ce
from relayspec.mapper_campaign import restore_mapper


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensors_sha(values):
    """Device-independent exact diagnostic digest, including tensor metadata."""
    digest = hashlib.sha256()
    for name, value in values:
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}:".encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    settings = config["adaptation_pilot"]
    count = settings["pilot_distinct_examples"]
    updates = settings["pilot_updates"]
    batch_size = settings["batch_size"]
    if not 1 <= count <= 2048 or updates <= 0 or batch_size != 4:
        raise ValueError(
            "bounded adaptation requires at most 2048 records and batch four"
        )
    record_device = settings.get("prepared_records_device", "cuda")
    if record_device not in {"cpu", "cuda"}:
        raise ValueError("unsupported prepared-record device")
    diagnostics = settings.get("diagnostics", False)
    if settings.get("deterministic_algorithms", False):
        if (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            != settings["cublas_workspace_config"]
        ):
            raise ValueError(
                "deterministic pilot requires its declared cuBLAS workspace"
            )
        torch.use_deterministic_algorithms(True)
    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4 or torch.cuda.device_count() != 4:
        raise ValueError("pilot requires four independent GPU workers")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    output = Path(os.environ["RELAYSPEC_FIT_OUTPUT"]) / f"rank{rank}"
    output.mkdir(parents=True, exist_ok=False)
    cache_root = Path(os.environ["RELAYSPEC_FEATURE_CACHE"])
    index_path = cache_root / "cache-index.json"
    index = json.loads(index_path.read_text())
    if (
        index["status"] != "pass"
        or sha(index_path) != settings["feature_cache_index_sha256"]
    ):
        raise ValueError("pilot requires its declared immutable feature cache")
    if any(
        index["metadata"]["config"][key] != config[key]
        for key in ("target", "proposer")
    ):
        raise ValueError("pilot models differ from frozen-feature models")
    mapper_path = Path(settings["initial_mapper"])
    if sha(mapper_path) != settings["initial_mapper_sha256"]:
        raise ValueError("initial mapper hash changed")
    checkpoint = torch.load(mapper_path, map_location="cpu", weights_only=True)
    model_cache = Path(os.environ["TRANSFORMERS_CACHE"])
    draft_class, _ = import_official_dflash(
        os.environ["DFLASH_SOURCE"], config["proposer"]["source_commit"]
    )

    def load_model(spec, cls=AutoModelForCausalLM):
        return (
            cls.from_pretrained(
                spec["id"],
                revision=spec["revision"],
                cache_dir=model_cache,
                attn_implementation="sdpa",
                dtype=torch.bfloat16,
                local_files_only=True,
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )

    started = time.perf_counter()
    target = load_model(config["target"])
    source = load_model(config["source_trunk"]["model"])
    draft = load_model(config["proposer"], draft_class)
    mapper, taps = restore_mapper(
        checkpoint,
        target_hidden_size=target.config.hidden_size,
        draft_hidden_size=draft.config.hidden_size,
        eps=target.config.rms_norm_eps,
    )
    if taps != tuple(index["metadata"]["target_layer_ids"]):
        raise ValueError("initial mapper taps differ from feature cache")
    mapper = mapper.to(device).eval().requires_grad_(False)
    model_load_seconds = time.perf_counter() - started
    preparation_started = time.perf_counter()
    block = 16
    records = []
    entries = [entry for entry in index["entries"] if entry["split"] == "train"][:count]
    if len(entries) != count or len({e["file"] for e in entries}) != count:
        raise ValueError("pilot requires its declared distinct cached records")
    with torch.no_grad():
        for entry in entries:
            path = cache_root / entry["file"]
            if sha(path) != entry["sha256"]:
                raise ValueError("cached record hash changed")
            saved = torch.load(path, weights_only=True, map_location="cpu")
            ids = saved["input_ids"].to(device)
            if ids.shape[1] <= block:
                raise ValueError("pilot record is shorter than its proposal block")
            prefix = ids.shape[1] - block
            labels = (
                target(ids, use_cache=False, logits_to_keep=block)
                .logits[:, :-1]
                .argmax(-1)
            )
            noise = torch.full(
                (1, block), int(draft.mask_token_id), device=device, dtype=torch.long
            )
            noise[:, 0] = ids[:, prefix]
            records.append(
                (
                    saved["x"][:, :prefix].to(record_device),
                    source.model.embed_tokens(noise).to(record_device),
                    torch.arange(ids.shape[1], device=record_device).unsqueeze(0),
                    labels.to(record_device),
                )
            )
    head = source.lm_head
    del source, target
    torch.cuda.empty_cache()
    preparation_seconds = time.perf_counter() - preparation_started
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    base_digest = frozen_base_digest(draft)

    def logits(record, model=draft):
        features, noise, positions = (value.to(device) for value in record[:3])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            context = model.hidden_norm(mapper(features))
            hidden = _conditioned_dflash_forward(
                model,
                conditioned_context=context,
                noise_embedding=noise,
                position_ids=positions,
                past_key_values=None,
            )
            return head(hidden[:, 1 - block :])

    with torch.no_grad():
        initial_logits = logits(records[0]).clone()
    fusion_equivalence = None
    if settings.get("check_direct_fusion_equivalence", False):
        inherited_fc = draft.fc
        comparisons = []
        try:
            draft.fc = mapper
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for record in records[:4]:
                    features, noise, positions = (
                        value.to(device) for value in record[:3]
                    )
                    fused_hidden = draft(
                        position_ids=positions,
                        noise_embedding=noise,
                        target_hidden=features,
                        past_key_values=None,
                        use_cache=True,
                        is_causal=False,
                    )
                    fused_logits = head(fused_hidden[:, 1 - block :])
                    comparisons.append(torch.equal(fused_logits, logits(record)))
        finally:
            draft.fc = inherited_fc
        if not all(comparisons):
            raise ValueError(
                "direct input-fusion placement differs from relay conditioning"
            )
        fusion_equivalence = {
            "status": "pass",
            "records": len(comparisons),
            "logits_bit_identical": True,
            "scope": "The same mapper is installed as the official drafter fc module. "
            "Implementation equivalence on four teacher-forced records, not a new competitor "
            "or full cached-decoding equivalence claim.",
        }
    ranks = settings["lora_ranks"]
    if len(ranks) != 2:
        raise ValueError("pilot requires two declared LoRA ranks")
    lora_rank = (0, ranks[0], ranks[1], ranks[1])[rank]
    if lora_rank:
        trainable = inject_lora(
            draft,
            tuple(settings["target_suffixes"]),
            lora_rank,
            settings["lora_alpha_multiplier"] * lora_rank,
        )
        with torch.no_grad():
            if not torch.equal(logits(records[0]), initial_logits):
                raise ValueError("zero-initialized LoRA changes the drafter")
        zero = export_merged_lora(
            draft,
            expected_base_digest=base_digest,
            proposer=config["proposer"],
            mapper_sha256=settings["initial_mapper_sha256"],
        )
        torch.save(zero, output / "zero-adaptation.pt")
    else:
        mapper.requires_grad_(True)
        trainable = list(mapper.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=settings["learning_rate"], weight_decay=0.0
    )
    diagnostic = {}
    if diagnostics:
        diagnostic = {
            "prepared_records_sha256": tensors_sha(
                (f"{i}/{j}", tensor)
                for i, record in enumerate(records)
                for j, tensor in enumerate(record)
            ),
            "labels_sha256": tensors_sha(
                (str(i), record[-1]) for i, record in enumerate(records)
            ),
            "initial_trainable_sha256": tensors_sha(
                (str(i), p) for i, p in enumerate(trainable)
            ),
            "initial_logits_sha256": tensors_sha([("logits", initial_logits)]),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }
        (output / "diagnostics.json").write_text(
            json.dumps(diagnostic, indent=2) + "\n"
        )
    setup_seconds = time.perf_counter() - started
    history = []
    torch.cuda.synchronize()
    loop_started = time.perf_counter()
    for step in range(updates):
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        if diagnostics and step == 0:
            initial_rng = torch.cuda.get_rng_state(device)
            initial_cpu_rng = torch.get_rng_state()
            microsteps = []
        batch = [records[(step * batch_size + i) % count] for i in range(batch_size)]
        for record in batch:
            prediction = logits(record)
            loss = greedy_agreement_ce(prediction, record[-1].to(device)) / batch_size
            loss.backward()
            loss_value += float(loss.detach())
            if diagnostics and step == 0:
                microsteps.append(
                    {
                        "logits_sha256": tensors_sha([("logits", prediction)]),
                        "accumulated_gradient_sha256": tensors_sha(
                            (str(i), p.grad) for i, p in enumerate(trainable)
                        ),
                    }
                )
        if diagnostics and step == 0:
            original_gradients = [p.grad.detach().clone() for p in trainable]
            post_rng = torch.cuda.get_rng_state(device)
            post_cpu_rng = torch.get_rng_state()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.set_rng_state(initial_rng, device)
            torch.set_rng_state(initial_cpu_rng)
            repeat_logits = []
            for record in batch:
                prediction = logits(record)
                repeat_logits.append(tensors_sha([("logits", prediction)]))
                (
                    greedy_agreement_ce(prediction, record[-1].to(device)) / batch_size
                ).backward()
            diagnostic["first_step"] = {
                "microsteps": microsteps,
                "repeat_logits_bit_identical": repeat_logits
                == [row["logits_sha256"] for row in microsteps],
                "repeat_gradients_bit_identical": all(
                    torch.equal(p.grad, original)
                    for p, original in zip(trainable, original_gradients, strict=True)
                ),
                "repeat_gradient_max_abs_difference": max(
                    (p.grad - original).abs().max().item()
                    for p, original in zip(trainable, original_gradients, strict=True)
                ),
            }
            for p, original in zip(trainable, original_gradients, strict=True):
                p.grad = original
            torch.cuda.set_rng_state(post_rng, device)
            torch.set_rng_state(post_cpu_rng)
            (output / "diagnostics.json").write_text(
                json.dumps(diagnostic, indent=2) + "\n"
            )
        if not all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in trainable
        ):
            raise ValueError("missing or nonfinite adaptation gradients")
        if not any(torch.count_nonzero(p.grad) for p in trainable):
            raise ValueError("all adaptation gradients are zero")
        optimizer.step()
        history.append(
            {
                "step": step + 1,
                "loss": loss_value,
                "loop_elapsed_seconds": time.perf_counter() - loop_started,
                "presentations": (step + 1) * batch_size,
            }
        )
    torch.cuda.synchronize()
    loop_seconds = time.perf_counter() - loop_started
    if frozen_base_digest(draft) != base_digest:
        raise ValueError("pilot updated inherited drafter parameters")
    export_gate = None
    if lora_rank:
        payload = export_merged_lora(
            draft,
            expected_base_digest=base_digest,
            proposer=config["proposer"],
            mapper_sha256=settings["initial_mapper_sha256"],
        )
        if all(
            torch.equal(payload["weights"][name], value)
            for name, value in zero["weights"].items()
        ):
            raise ValueError("LoRA training produced no merged weight change")
        path = output / "adaptation.pt"
        torch.save(payload, path)
        reloaded = load_model(config["proposer"], draft_class)
        apply_merged_lora(
            reloaded,
            torch.load(path, map_location="cpu", weights_only=True),
            proposer=config["proposer"],
            mapper_sha256=settings["initial_mapper_sha256"],
        )
        merged = copy.deepcopy(draft)
        merge_lora_into_base(merged)
        with torch.no_grad():
            if not torch.equal(
                logits(records[0], merged), logits(records[0], reloaded)
            ):
                raise ValueError("merged drafter reload differs from merged reference")
        export_gate = {
            "status": "pass",
            "merged_reload_bit_identical": True,
            "zero_update_bit_identical": True,
            "checkpoint_sha256": sha(path),
        }
        del merged, reloaded
    else:
        checkpoint["relay"] = {
            name: value.detach().cpu() for name, value in mapper.state_dict().items()
        }
        checkpoint["adaptation_pilot"] = {
            "objective": "teacher_forced_greedy_ce",
            "additional_updates": updates,
        }
        torch.save(checkpoint, output / "mapper.pt")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "updates": updates,
            "lora_rank": lora_rank,
        },
        output / "optimizer.pt",
    )
    training_model = draft if lora_rank else mapper
    training_state = {
        "format": "relayspec-adaptation-training-state-v1",
        "trainable": {
            name: p.detach().cpu().clone()
            for name, p in training_model.named_parameters()
            if p.requires_grad
        },
        "optimizer": optimizer.state_dict(),
        "updates": updates,
        "lora_rank": lora_rank,
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device),
        "config_sha256": sha(args.config),
        "initial_mapper_sha256": settings["initial_mapper_sha256"],
        "feature_cache_index_sha256": sha(index_path),
        "ordered_record_files": [entry["file"] for entry in entries],
    }
    torch.save(training_state, output / "training-state.pt")
    reloaded_state = torch.load(
        output / "training-state.pt", weights_only=True, map_location="cpu"
    )
    if any(
        not torch.equal(value, reloaded_state["trainable"][name])
        for name, value in training_state["trainable"].items()
    ):
        raise ValueError("trainable checkpoint serialization changed values")
    (output / "training.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in history)
    )
    (output / "adaptation-fit-gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "rank": rank,
                "lora_rank": lora_rank,
                "updates": updates,
                "distinct_examples": min(count, updates * batch_size),
                "prepared_distinct_examples": count,
                "presentations": updates * batch_size,
                "batch_size": batch_size,
                "prepared_records_device": record_device,
                "ordered_record_files": [entry["file"] for entry in entries],
                "training_state_sha256": sha(output / "training-state.pt"),
                "trainable_parameters": sum(p.numel() for p in trainable),
                "setup_seconds": setup_seconds,
                "model_load_seconds": model_load_seconds,
                "teacher_and_feature_preparation_seconds": preparation_seconds,
                "loop_seconds": loop_seconds,
                "total_seconds": time.perf_counter() - started,
                "feature_cache_index_sha256": sha(index_path),
                "initial_mapper_sha256": settings["initial_mapper_sha256"],
                "inherited_drafter_sha256": base_digest,
                "inherited_weights_unchanged": True,
                "export": export_gate,
                "direct_fusion_equivalence": fusion_equivalence,
                "config_sha256": sha(args.config),
                "scope": "Resource/correctness pilot. Teacher-forced target greedy labels on the last 15 proposal positions. Separate rank0 connector-only CE control and ranks1/2/3 frozen-connector drafter LoRA. Ranks2/3 are duplicate-seed checks. No full matched-compute or task-quality conclusion.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
