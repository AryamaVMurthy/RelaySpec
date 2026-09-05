from __future__ import annotations

import argparse
import hashlib
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

from relayspec.benchmarking import resolve_revision
from relayspec.dflash import import_official_dflash
from relayspec.eagle3 import import_official_eagle3
from relayspec.feature_cache import finalize_cache, write_cache_shard
from relayspec.fitting_validation import interface_diagnostics, validate_fit_split
from relayspec.generation import _conditioned_dflash_forward
from relayspec.losses import (
    expected_accepted_length_surrogate,
    explicit_l2_penalty,
    greedy_agreement_ce,
    hard_accepted_prefix_diagnostic,
    interface_alignment_loss,
    proposal_kl_loss,
)
from relayspec.proposers import project_source_interface
from relayspec.relay import TargetFeatureRelay, extract_hidden_taps
from relayspec.training_controls import (
    save_training_checkpoint,
    training_data_accounting,
)
from relayspec.vocab_bridge import (
    align_positions,
    canonicalize_offsets,
    content_windows,
    filter_offsets_to_windows,
)


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


def encode_example_with_offsets(
    tokenizer: Any,
    row: dict[str, Any],
    max_length: int,
    device: torch.device,
) -> tuple[torch.LongTensor, list[tuple[int, int]], str]:
    """Like `encode_example`, but also returns each token's character span.

    `apply_chat_template` does not itself support
    `return_offsets_mapping`, so the template is rendered to a plain string
    first (with its own special tokens included as literal text, which is
    what every standard HF chat template does), then tokenized directly
    with offsets. Re-tokenizing the rendered string with
    `add_special_tokens=False` reproduces the same ids `encode_example`
    would give, since the template's special tokens are already literal
    text in the rendered string.
    """
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": row["problem"]},
            {"role": "assistant", "content": row["solution"]},
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"][:, :max_length].to(device)
    offsets = [
        tuple(pair) for pair in encoded["offset_mapping"][0, :max_length].tolist()
    ]
    return input_ids, offsets, rendered


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
            revision=resolve_revision(config.target.id, config.target.revision),
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
        revision=resolve_revision(config.target.id, config.target.revision),
        cache_dir=cache_dir,
        local_files_only=True,
    )
    # X1: fitting a relay where the target does not share the source's
    # tokenizer (see
    # docs/plans/2026-09-02-relayspec-source-free-adaptive-relay-plan.md
    # section 3.5b). The ordinary loop feeds one token sequence to both
    # models, which only makes sense when they share a vocabulary. Here each
    # model instead encodes the same source text independently, and the
    # per-position regression collapses to a single last-position
    # comparison, since two different tokenizers give no general per-position
    # alignment between the two sequences.
    cross_family = bool(training.get("cross_family", False))
    source_tokenizer = None
    if cross_family:
        if str(training.get("supervision", "source_interface")) != "source_interface":
            raise ValueError(
                "cross_family fitting only supports source_interface supervision"
            )
        if float(training.get("proposal_kl_weight", 0.0)) > 0:
            raise ValueError("cross_family fitting does not support proposal_kl_weight")
        source_tokenizer = AutoTokenizer.from_pretrained(
            source_config.model.id,
            revision=source_config.model.revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    target_layer_ids = tuple(int(value) for value in training["target_layer_ids"])
    if len(target_layer_ids) != len(draft.target_layer_ids):
        raise ValueError("target relay tap count must match the trained proposer")
    cache_settings = training.get("feature_cache")
    if cache_settings:
        if (
            cross_family
            or training.get("supervision", "source_interface") != "source_interface"
            or training.get("proposal_kl_weight", 0)
        ):
            raise ValueError(
                "feature caching currently supports pure same-tokenizer interface fitting"
            )
        root = Path(os.environ["RELAYSPEC_FEATURE_CACHE"])
        root.mkdir(parents=True, exist_ok=True)
        fit_path = Path(training["manifest_path"])
        val_path = Path(training["validation_manifest_path"])
        fit_rows = json.loads(fit_path.read_text())["records"]
        val_rows = json.loads(val_path.read_text())["records"]
        fit_rows = fit_rows[: int(cache_settings["train_records"])]
        val_rows = val_rows[: int(cache_settings["validation_records"])]
        if len(fit_rows) != int(cache_settings["train_records"]) or len(
            val_rows
        ) != int(cache_settings["validation_records"]):
            raise ValueError("feature cache budget exceeds available records")
        split = validate_fit_split(fit_rows, val_rows)
        groups = {"train": fit_rows, "validation": val_rows}
        cache_started = time.perf_counter()

        @torch.no_grad()
        def extract_cache_row(row):
            ids = encode_example(tokenizer, row, int(training["max_length"]), device)
            source_output = source(
                ids, use_cache=False, output_hidden_states=True, logits_to_keep=1
            )
            teacher = project_source_interface(
                draft,
                extract_hidden_taps(
                    source_output.hidden_states,
                    tuple(int(i) for i in draft.target_layer_ids),
                ),
                family=proposer_family,
            )
            del source_output
            target_output = target(
                ids, use_cache=False, output_hidden_states=True, logits_to_keep=1
            )
            features = extract_hidden_taps(
                target_output.hidden_states, target_layer_ids
            )
            del target_output
            return features, teacher, ids

        write_cache_shard(
            root, groups, rank=rank, world_size=world_size, extract=extract_cache_row
        )
        dist.barrier()
        if rank == 0:
            output_norm = None
            if proposer_family == "dflash":
                norm = draft.hidden_norm
                if type(norm).__name__ != "Qwen3RMSNorm":
                    raise ValueError(
                        "cache consumer needs an explicit implementation for this output norm"
                    )
                output_norm = {
                    "kind": "Qwen3RMSNorm",
                    "eps": norm.variance_epsilon,
                    "weight": norm.weight.detach().cpu(),
                }
            torch.save(output_norm, root / "output-norm.pt")
            metadata = {
                "config": payload,
                "relay_training": training,
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "manifest_sha256": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [fit_path, val_path]
                },
                "target_layer_ids": target_layer_ids,
                "target_hidden_size": target.config.hidden_size,
                "draft_hidden_size": draft.config.hidden_size,
                "rms_norm_eps": target.config.rms_norm_eps,
                "family": proposer_family,
                "split": split,
                "output_norm_sha256": hashlib.sha256(
                    (root / "output-norm.pt").read_bytes()
                ).hexdigest(),
                "torch_version": str(torch.__version__),
                "elapsed_seconds": time.perf_counter() - cache_started,
                "normalization": "Raw target taps; teacher includes the released proposer interface. DFlash predictions must use output-norm.pt; EAGLE-3 predictions use identity.",
            }
            result = finalize_cache(
                root,
                counts={k: len(v) for k, v in groups.items()},
                world_size=world_size,
                metadata=metadata,
            )
            output_dir = Path(os.environ["RELAYSPEC_OUTPUT"])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "feature-cache-gate.json").write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "cache_root": str(root),
                        "counts": result["counts"],
                        "total_bytes": result["total_bytes"],
                        "total_tokens": result["total_tokens"],
                        "cache_index_sha256": hashlib.sha256(
                            (root / "cache-index.json").read_bytes()
                        ).hexdigest(),
                        "metadata": metadata,
                    },
                    indent=2,
                )
                + "\n"
            )
        dist.barrier()
        dist.destroy_process_group()
        return
    relay_architecture = str(training.get("relay_architecture", "normalized_linear"))
    if relay_architecture not in {"normalized_linear", "scale_preserving_linear"}:
        raise ValueError(f"unsupported relay architecture: {relay_architecture}")
    # Rigor-pass ablation: is a single linear map the right hypothesis
    # class? Only supported on the plain fitting path (not the delta/adapter
    # mechanisms, which are separate experiments with their own scope).
    mlp_hidden_width = training.get("mlp_hidden_width")
    mlp_hidden_width = int(mlp_hidden_width) if mlp_hidden_width is not None else None
    # E4-P1: freeze a relay already fit for one target and train only a small
    # adapter that maps a *different* target's raw tap width into that frozen
    # relay's native input space. See
    # docs/plans/2026-09-02-relayspec-source-free-adaptive-relay-plan.md
    # section 3.5, P1. Mutually exclusive with the ordinary warm-start
    # mechanism below: a frozen base is a fixed decoder, not a starting point.
    factorized_rank = training.get("factorized_rank")
    l2_weight = float(training.get("l2_weight", 0.0))
    if l2_weight and float(training["weight_decay"]):
        raise ValueError("controlled L2 and AdamW weight decay must be separate arms")
    freeze_base_relay = bool(training.get("freeze_base_relay", False))
    # E3 Stage B: freeze a relay already fit for one target and train only a
    # small low-rank delta on top of it, for a fine-tuned descendant that
    # keeps the same architecture and widths as the original target. See
    # docs/plans/2026-09-02-relayspec-source-free-adaptive-relay-plan.md
    # section 3.4. Unlike freeze_base_relay (E4-P1), no adapter is needed
    # here because the descendant's tap width matches the base relay's own.
    delta_rank = training.get("delta_rank")
    delta_rank = int(delta_rank) if delta_rank is not None else None
    # Residual nonlinear correction ablation: c = R z + U sigma(V z), a
    # GELU inserted between the delta's down- and up-projection, keeping
    # the frozen linear R as the backbone. Only meaningful alongside
    # delta_rank.
    delta_nonlinear = bool(training.get("delta_nonlinear", False))
    initial_checkpoint = os.environ.get(
        "RELAYSPEC_INITIAL_CHECKPOINT",
        training.get("initial_checkpoint_path"),
    )
    if (
        sum([freeze_base_relay, delta_rank is not None, initial_checkpoint is not None])
        > 1
    ):
        raise ValueError(
            "freeze_base_relay, delta_rank, and initial_checkpoint_path are "
            "alternative mechanisms; do not combine them"
        )
    if (mlp_hidden_width is not None or factorized_rank is not None) and (
        freeze_base_relay or delta_rank is not None
    ):
        raise ValueError("capacity maps cannot be combined with frozen-base adapters")
    if delta_rank is not None:
        delta_checkpoint_path = training["delta_base_checkpoint_path"]
        delta_checkpoint = torch.load(
            delta_checkpoint_path, map_location="cpu", weights_only=True
        )
        delta_base_target_layer_ids = tuple(
            int(value) for value in delta_checkpoint["target_layer_ids"]
        )
        if delta_base_target_layer_ids != target_layer_ids:
            raise ValueError(
                "delta base relay's target taps must exactly match the "
                "descendant's, since a delta assumes identical widths"
            )
        relay = TargetFeatureRelay(
            target_hidden_size=target.config.hidden_size,
            num_taps=len(target_layer_ids),
            draft_hidden_size=draft.config.hidden_size,
            eps=target.config.rms_norm_eps,
            normalize_input=str(
                delta_checkpoint.get("relay_architecture", "normalized_linear")
            )
            == "normalized_linear",
            delta_rank=delta_rank,
            delta_nonlinear=delta_nonlinear,
        ).to(device)
        relay.load_state_dict(delta_checkpoint["relay"], strict=False)
        relay.projection.weight.requires_grad_(False)
        adapter_parameters = list(relay.delta_down.parameters()) + list(
            relay.delta_up.parameters()
        )
        relay_architecture = str(
            delta_checkpoint.get("relay_architecture", "normalized_linear")
        )
        adapter_input_width = None
    elif freeze_base_relay:
        base_checkpoint_path = training["adapter_base_checkpoint_path"]
        base_checkpoint = torch.load(
            base_checkpoint_path, map_location="cpu", weights_only=True
        )
        base_target_layer_ids = tuple(
            int(value) for value in base_checkpoint["target_layer_ids"]
        )
        if len(base_target_layer_ids) != len(target_layer_ids):
            raise ValueError(
                "frozen base relay tap count must match the adapted target's tap count"
            )
        base_input_width = base_checkpoint["relay"]["projection.weight"].shape[1]
        base_target_hidden_size = base_input_width // len(base_target_layer_ids)
        adapter_input_width = target.config.hidden_size * len(target_layer_ids)
        relay = TargetFeatureRelay(
            target_hidden_size=base_target_hidden_size,
            num_taps=len(base_target_layer_ids),
            draft_hidden_size=draft.config.hidden_size,
            eps=target.config.rms_norm_eps,
            normalize_input=str(
                base_checkpoint.get("relay_architecture", "normalized_linear")
            )
            == "normalized_linear",
            adapter_input_width=adapter_input_width,
        ).to(device)
        relay.load_state_dict(base_checkpoint["relay"], strict=False)
        relay.projection.weight.requires_grad_(False)
        adapter_parameters = list(relay.adapter.parameters())
        relay_architecture = str(
            base_checkpoint.get("relay_architecture", "normalized_linear")
        )
    else:
        adapter_input_width = None
        relay = TargetFeatureRelay(
            target_hidden_size=target.config.hidden_size,
            num_taps=len(target_layer_ids),
            draft_hidden_size=draft.config.hidden_size,
            eps=target.config.rms_norm_eps,
            normalize_input=relay_architecture == "normalized_linear",
            mlp_hidden_width=mlp_hidden_width,
            factorized_rank=factorized_rank,
        ).to(device)
        if initial_checkpoint is not None:
            initial = torch.load(
                initial_checkpoint, map_location="cpu", weights_only=True
            )
            if (
                tuple(int(value) for value in initial["target_layer_ids"])
                != target_layer_ids
            ):
                raise ValueError(
                    "initial relay target taps do not match current training"
                )
            relay.load_state_dict(initial["relay"], strict=True)
        adapter_parameters = None
    distributed = DistributedDataParallel(relay, device_ids=[local_rank])
    adapter_optimizer = str(training.get("adapter_optimizer", "adamw"))
    if adapter_optimizer not in {"adamw", "sgd"}:
        raise ValueError(f"unsupported adapter optimizer: {adapter_optimizer}")
    trainable_parameters = (
        adapter_parameters
        if adapter_parameters is not None
        else distributed.parameters()
    )
    if adapter_parameters is not None and adapter_optimizer == "sgd":
        # A full adapter (e.g. mapping one target's raw tap width into
        # another's) can be as large as the base relay's own projection, so
        # AdamW's two extra per-parameter buffers can OOM a 48 GB card
        # already holding the larger target model. SGD-with-momentum halves
        # that optimizer-state memory; it is a standard, adequate fallback
        # for training a single linear layer when AdamW does not fit.
        optimizer = torch.optim.SGD(
            trainable_parameters,
            lr=float(training["learning_rate"]),
            momentum=0.9,
            weight_decay=float(training["weight_decay"]),
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )

    records = json.loads(Path(training["manifest_path"]).read_text(encoding="utf-8"))[
        "records"
    ]
    steps = int(training["steps"])
    accounting = training_data_accounting(records, training, world_size)
    local_rows = records[rank::world_size]
    checkpoint_steps = set(int(x) for x in training.get("checkpoint_steps", []))
    if any(x <= 0 or x > steps for x in checkpoint_steps):
        raise ValueError("checkpoint steps must lie within the training run")
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

    # Axis A/B: verifier-supervised source-free retargeting. See
    # docs/plans/2026-09-02-relayspec-source-free-adaptive-relay-plan.md
    # section 3.1. Default ("source_interface") reproduces the exact prior
    # behaviour above, byte for byte: it never enters any branch below.
    supervision = str(training.get("supervision", "source_interface"))
    if supervision not in {"source_interface", "target_verifier"}:
        raise ValueError(f"unsupported supervision mode: {supervision}")
    verifier_objective = str(training.get("verifier_objective", "greedy_agreement_ce"))
    if verifier_objective not in {
        "greedy_agreement_ce",
        "proposal_kl_to_target",
        "accepted_prefix_surrogate",
    }:
        raise ValueError(f"unsupported verifier objective: {verifier_objective}")
    regression_anchor_weight = float(training.get("regression_anchor_weight", 0.0))
    if regression_anchor_weight < 0:
        raise ValueError("regression anchor weight must be non-negative")
    verifier_block_size = int(
        training.get("verifier_block_size", getattr(draft, "block_size", 2))
    )
    if supervision == "target_verifier":
        if proposer_family != "dflash":
            raise ValueError(
                "target-verifier supervision is currently DFlash-specific, "
                "pending the EAGLE-3 differentiable conditioned forward "
                "(implementation plan Task 4)"
            )
        if verifier_block_size < 2:
            raise ValueError(
                "verifier-supervised training requires block size at least two"
            )
        if proposal_weight > 0:
            raise ValueError(
                "proposal_kl_weight and supervision=target_verifier are "
                "alternative mechanisms for the same block; do not combine them"
            )
    # Warm start (Section 3.2 mitigation 1) reuses the existing
    # initial_checkpoint_path/RELAYSPEC_INITIAL_CHECKPOINT mechanism defined
    # above; no separate config key is needed for it.
    output_dir = Path(os.environ["RELAYSPEC_OUTPUT"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"relay-train-rank{rank}.jsonl"
    checkpoint_dir = (
        Path(os.environ["RELAYSPEC_CACHE_DIR"])
        / "relayspec"
        / "checkpoints"
        / config.run_name
    )
    checkpoint_seconds = 0.0

    def checkpoint_payload(saved_step: int) -> dict:
        return {
            "relay": distributed.module.state_dict(),
            "target_layer_ids": target_layer_ids,
            "source_layer_ids": tuple(int(value) for value in draft.target_layer_ids),
            "steps": saved_step,
            "seed": config.seed,
            "data_accounting": {
                **accounting,
                "record_presentations": saved_step * world_size,
                "distinct_records_seen": min(
                    saved_step * world_size, accounting["distinct_records"]
                ),
            },
            "feature_objective": feature_objective,
            "weight_decay": float(training["weight_decay"]),
            "gradient_clip": training.get("gradient_clip"),
            "initialized_from": (
                training.get("delta_base_checkpoint_path")
                if delta_rank is not None
                else training.get("adapter_base_checkpoint_path")
                if freeze_base_relay
                else initial_checkpoint
            ),
            "proposer_family": proposer_family,
            "relay_architecture": relay_architecture,
            "adapter_input_width": adapter_input_width,
            "delta_rank": delta_rank,
            "delta_nonlinear": delta_nonlinear,
            "mlp_hidden_width": mlp_hidden_width,
            "factorized_rank": factorized_rank,
            "l2_weight": l2_weight,
            "mapper_parameters": sum(p.numel() for p in relay.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in relay.parameters() if p.requires_grad
            ),
            "supervision": supervision,
            "verifier_objective": (
                verifier_objective if supervision == "target_verifier" else None
            ),
            "regression_anchor_weight": regression_anchor_weight,
        }

    validation_cache = {}
    validation_path = training.get("validation_manifest_path")
    if validation_path:
        if cross_family or supervision != "source_interface" or proposal_weight:
            raise ValueError(
                "fitting diagnostics currently require same-tokenizer pure interface fitting"
            )
        validation_rows = json.loads(Path(validation_path).read_text())["records"]
        split = validate_fit_split(records, validation_rows)
        diagnostic_count = int(training.get("diagnostic_train_examples", 32))
        if diagnostic_count < world_size or len(validation_rows) < world_size:
            raise ValueError("diagnostic splits must cover every worker")
        groups = {"train": records[:diagnostic_count], "validation": validation_rows}
        with torch.no_grad():
            for name, group in groups.items():
                validation_cache[name] = []
                for row in group[rank::world_size]:
                    ids = encode_example(
                        tokenizer, row, int(training["max_length"]), device
                    )
                    source_output = source(
                        ids,
                        use_cache=False,
                        output_hidden_states=True,
                        logits_to_keep=1,
                    )
                    context = project_source_interface(
                        draft,
                        extract_hidden_taps(
                            source_output.hidden_states,
                            tuple(int(v) for v in draft.target_layer_ids),
                        ),
                        family=proposer_family,
                    )
                    del source_output
                    target_output = target(
                        ids,
                        use_cache=False,
                        output_hidden_states=True,
                        logits_to_keep=1,
                    )
                    features = extract_hidden_taps(
                        target_output.hidden_states, target_layer_ids
                    )
                    validation_cache[name].append((features.cpu(), context.cpu()))
                    del target_output, features, context
        if rank == 0:
            (output_dir / "validation-split.json").write_text(
                json.dumps(split, indent=2) + "\n"
            )
    validation_seconds = 0.0

    def record_validation(saved_step):
        if not validation_cache:
            return
        result = {
            name: interface_diagnostics(
                relay,
                examples,
                device=device,
                objective=feature_objective,
                historical_cosine_weight=historical_cosine_weight,
                output_transform=draft.hidden_norm
                if proposer_family == "dflash"
                else None,
            )
            for name, examples in validation_cache.items()
        }
        with (output_dir / f"fitting-validation-rank{rank}.jsonl").open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "step": saved_step,
                        "rank": rank,
                        "groups": result,
                        "weighting": "mean per record; reduce workers using record counts",
                        "regularization_included": False,
                    }
                )
                + "\n"
            )

    record_validation(0)
    started = time.perf_counter()
    with metrics_path.open("x", encoding="utf-8") as metrics:
        iterator = islice(cycle(local_rows), steps)
        for step, row in enumerate(iterator, start=1):
            target_offsets: list[tuple[int, int]] | None = None
            source_offsets: list[tuple[int, int]] | None = None
            if cross_family:
                input_ids, target_offsets, target_rendered = (
                    encode_example_with_offsets(
                        tokenizer, row, int(training["max_length"]), device
                    )
                )
                source_ids, source_offsets, source_rendered = (
                    encode_example_with_offsets(
                        source_tokenizer, row, int(training["max_length"]), device
                    )
                )
            else:
                input_ids = encode_example(
                    tokenizer,
                    row,
                    int(training["max_length"]),
                    device,
                )
                source_ids = None
            feature_ids = input_ids
            if supervision == "target_verifier":
                if input_ids.shape[1] <= verifier_block_size:
                    raise ValueError("verifier example is shorter than its draft block")
                feature_ids = input_ids[:, :-verifier_block_size]
            elif proposal_weight > 0:
                if input_ids.shape[1] <= proposal_block_size:
                    raise ValueError("proposal example is shorter than its draft block")
                feature_ids = input_ids[:, :-proposal_block_size]
            optimizer.zero_grad(set_to_none=True)
            # A source-interface regression is computed whenever it is the
            # trained objective, or whenever it is only a diagnostic anchor
            # (Section 3.2 mitigation 2/3). Under pure target_verifier
            # supervision with a zero anchor weight, this stays False and
            # `source` is never given a sequence forward: only its embedding
            # and LM head matrices are read below, which are single-matrix
            # ops belonging to the released proposer contract, not a trunk
            # forward pass (plan section 1.1).
            compute_source_regression = (
                supervision == "source_interface" or regression_anchor_weight > 0.0
            )
            with torch.no_grad():
                teacher_context = None
                source_features = None
                if compute_source_regression:
                    source_output = source(
                        source_ids if cross_family else feature_ids,
                        use_cache=False,
                        output_hidden_states=True,
                        logits_to_keep=1,
                    )
                    source_features = extract_hidden_taps(
                        source_output.hidden_states,
                        tuple(int(value) for value in draft.target_layer_ids),
                    )
                    del source_output
                target_input = (
                    input_ids if supervision == "target_verifier" else feature_ids
                )
                target_logits_to_keep = (
                    verifier_block_size if supervision == "target_verifier" else 1
                )
                target_output = target(
                    target_input,
                    use_cache=False,
                    output_hidden_states=True,
                    logits_to_keep=target_logits_to_keep,
                )
                target_features = extract_hidden_taps(
                    target_output.hidden_states,
                    target_layer_ids,
                )
                if cross_family:
                    # Two tokenizers give no shared token sequence, so the
                    # ordinary "same position in both" pairing does not
                    # apply. Align by character span instead (Step 2,
                    # docs/plans/2026-09-03-relayspec-cross-family-fix-plan.md):
                    # every target position is paired with whichever source
                    # position covers the same text, recovering real
                    # per-position supervision instead of collapsing the
                    # whole example to its last position.
                    assert source_offsets is not None and target_offsets is not None
                    # Restrict to the problem/solution text itself: the two
                    # models' chat templates render different boilerplate
                    # (system prompt, date stamp, role markers) around the
                    # same content, so aligning the full rendered strings
                    # would pair unrelated template tokens with each other.
                    source_windows = content_windows(
                        source_rendered, row["problem"], row["solution"]
                    )
                    target_windows = content_windows(
                        target_rendered, row["problem"], row["solution"]
                    )
                    source_keep, source_offsets_filtered = filter_offsets_to_windows(
                        source_offsets, source_windows
                    )
                    target_keep, target_offsets_filtered = filter_offsets_to_windows(
                        target_offsets, target_windows
                    )
                    # The two filtered offset lists are still in each
                    # tokenizer's own rendered string's coordinates, which
                    # differ because the template boilerplate before each
                    # window has a different length in each rendering.
                    # Put both on the same content-only axis before
                    # comparing spans.
                    source_offsets_canonical = canonicalize_offsets(
                        source_offsets_filtered, source_windows
                    )
                    target_offsets_canonical = canonicalize_offsets(
                        target_offsets_filtered, target_windows
                    )
                    alignment = align_positions(
                        source_offsets_canonical, target_offsets_canonical
                    )
                    target_index = [
                        target_keep[t] for t, s in enumerate(alignment) if s is not None
                    ]
                    source_index = [source_keep[s] for s in alignment if s is not None]
                    if not target_index:
                        raise ValueError(
                            "cross_family alignment found no overlapping positions"
                        )
                    index_t = torch.tensor(target_index, device=device)
                    index_s = torch.tensor(source_index, device=device)
                    target_features = target_features.index_select(1, index_t)
                    if source_features is not None:
                        source_features = source_features.index_select(1, index_s)
                else:
                    target_features = target_features[:, : feature_ids.shape[1], :]
                target_features = target_features.detach()
                if compute_source_regression:
                    assert source_features is not None
                    teacher_context = project_source_interface(
                        draft,
                        source_features,
                        family=proposer_family,
                    ).detach()
                verifier_logits = None
                verifier_tokens = None
                if supervision == "target_verifier":
                    # target_output.logits holds the last verifier_block_size
                    # positions of target_input. Dropping the final one keeps
                    # exactly the (verifier_block_size - 1) causal positions
                    # feature_ids.shape[1] .. feature_ids.shape[1]+block-2,
                    # which predict the same block tokens 1..block-1 that
                    # predicted_logits below is sliced to, since block
                    # position 0 is given as the real committed token, not
                    # predicted.
                    verifier_logits = target_output.logits[:, :-1, :].detach()
                    verifier_tokens = verifier_logits.argmax(dim=-1)
                del target_output
                teacher_logits = None
                noise_embedding = None
                position_ids = None
                block_size_for_conditioning = (
                    verifier_block_size
                    if supervision == "target_verifier"
                    else proposal_block_size
                )
                if supervision == "target_verifier" or proposal_weight > 0:
                    block_ids = torch.full(
                        (1, block_size_for_conditioning),
                        int(draft.mask_token_id),
                        dtype=torch.long,
                        device=device,
                    )
                    block_ids[:, 0] = input_ids[:, feature_ids.shape[1]]
                    noise_embedding = source.model.embed_tokens(block_ids)
                    position_ids = torch.arange(
                        feature_ids.shape[1] + block_size_for_conditioning,
                        device=device,
                    ).unsqueeze(0)
                if proposal_weight > 0:
                    # supervision == "source_interface" is enforced above
                    # whenever proposal_weight > 0, so source_features was
                    # already computed for the regression teacher and is
                    # reused here rather than forwarding the source a second
                    # time.
                    assert source_features is not None
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
                del source_features
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predicted_context = distributed(target_features)
                if proposer_family == "dflash":
                    predicted_context = draft.hidden_norm(predicted_context)
            if teacher_context is not None:
                mse = F.mse_loss(predicted_context.float(), teacher_context.float())
                cosine = (
                    1.0
                    - F.cosine_similarity(
                        predicted_context.float(),
                        teacher_context.float(),
                        dim=-1,
                    ).mean()
                )
            else:
                mse = torch.zeros((), device=device)
                cosine = torch.zeros((), device=device)
            verifier_loss = torch.zeros((), device=device)
            hard_diag: dict[str, torch.Tensor] | None = None
            if supervision == "target_verifier":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predicted_hidden = _conditioned_dflash_forward(
                        draft,
                        conditioned_context=predicted_context,
                        noise_embedding=noise_embedding,
                        position_ids=position_ids,
                        past_key_values=None,
                    )
                    predicted_logits = source.lm_head(
                        predicted_hidden[:, 1 - verifier_block_size :, :]
                    )
                if verifier_objective == "greedy_agreement_ce":
                    verifier_loss = greedy_agreement_ce(
                        predicted_logits, verifier_tokens
                    )
                elif verifier_objective == "proposal_kl_to_target":
                    verifier_loss = proposal_kl_loss(
                        predicted_logits,
                        verifier_logits,
                        temperature=proposal_temperature,
                    )
                else:
                    verifier_loss = expected_accepted_length_surrogate(
                        predicted_logits, verifier_tokens
                    )
                hard_diag = hard_accepted_prefix_diagnostic(
                    predicted_logits.detach(), verifier_tokens
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
            if supervision == "target_verifier":
                anchor_loss = torch.zeros((), device=device)
                if regression_anchor_weight > 0.0:
                    assert teacher_context is not None
                    anchor_loss = interface_alignment_loss(
                        predicted_context,
                        teacher_context,
                        objective=feature_objective,
                        historical_cosine_weight=historical_cosine_weight,
                    )
                feature_loss = verifier_loss + regression_anchor_weight * anchor_loss
            else:
                feature_loss = interface_alignment_loss(
                    predicted_context,
                    teacher_context,
                    objective=feature_objective,
                    historical_cosine_weight=historical_cosine_weight,
                )
            l2_penalty = explicit_l2_penalty(distributed.parameters(), l2_weight)
            loss = feature_loss + proposal_weight * proposal_kl + l2_penalty
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
            if step in checkpoint_steps:
                validation_started = time.perf_counter()
                record_validation(step)
                torch.cuda.synchronize()
                validation_seconds += time.perf_counter() - validation_started
                torch.cuda.synchronize()
                save_started = time.perf_counter()
                dist.barrier()
                if rank == 0:
                    save_training_checkpoint(
                        checkpoint_dir / f"step-{step:06d}.pt",
                        checkpoint_payload(step),
                        optimizer,
                        time.perf_counter()
                        - started
                        - checkpoint_seconds
                        - validation_seconds,
                    )
                dist.barrier()
                checkpoint_seconds += time.perf_counter() - save_started
            hard_accepted_length_mean = None
            agreement_mass_mean = None
            if hard_diag is not None:
                hard_accepted_length_mean = float(
                    hard_diag["hard_accepted_length"].mean().item()
                )
                agreement_mass_mean = float(hard_diag["agreement_mass"].mean().item())
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
                "l2_penalty": float(l2_penalty.detach().item()),
                "l2_weight": l2_weight,
                "weight_decay": float(training["weight_decay"]),
                "gradient_norm": float(gradient_norm.detach().item()),
                "elapsed_seconds": time.perf_counter() - started,
                "initialized_from": initial_checkpoint,
                "proposer_family": proposer_family,
                "relay_architecture": relay_architecture,
                "supervision": supervision,
                "verifier_objective": (
                    verifier_objective if supervision == "target_verifier" else None
                ),
                "verifier_loss": float(verifier_loss.detach().item()),
                "regression_anchor_weight": regression_anchor_weight,
                "hard_accepted_length_mean": hard_accepted_length_mean,
                "agreement_mass_mean": agreement_mass_mean,
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
                verifier_loss,
            )
            if proposal_weight > 0:
                del predicted_hidden, predicted_logits, teacher_logits
            if supervision == "target_verifier":
                del predicted_hidden, predicted_logits, verifier_logits, verifier_tokens
                if hard_diag is not None:
                    del hard_diag

    dist.barrier()
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "relay.pt"
        torch.save(checkpoint_payload(steps), checkpoint_path)
        summary = {
            "status": "complete",
            "checkpoint": str(checkpoint_path),
            "intermediate_checkpoints": [
                str(checkpoint_dir / f"step-{x:06d}.pt")
                for x in sorted(checkpoint_steps)
            ],
            "data_accounting": accounting,
            "checkpoint_seconds": checkpoint_seconds,
            "validation_seconds": validation_seconds,
            "training_seconds_excluding_checkpoint_and_validation": time.perf_counter()
            - started
            - checkpoint_seconds
            - validation_seconds,
            "training_seconds_excluding_checkpoint_io": time.perf_counter()
            - started
            - checkpoint_seconds,
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
