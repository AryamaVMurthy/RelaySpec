from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import random
import statistics
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from relayspec.benchmarking import (
    aggregate_rows,
    benchmark_turns,
    generation_call_counts,
    generation_timings,
    model_load_plan,
    optional_generation_stats,
    resolve_revision,
    rotate_methods,
    shard_records,
)
from relayspec.dflash import import_official_dflash
from relayspec.drafter_adaptation import apply_merged_lora
from relayspec.generation import (
    cross_family_relay_dflash_generate,
    matched_full_target_dflash_generate,
    native_autoregressive_generate,
    relay_dflash_generate,
)
from relayspec.mapper_campaign import campaign_model_methods, restore_mapper
from relayspec.profiling import CudaRegionRecorder
from relayspec.relay import TargetFeatureRelay
from relayspec.sampling_rng import initialize_sampling_rng
from relayspec.sequence_scaling import validate_exact_input
from relayspec.source import SourceTapProvider
from relayspec.unfitted_controls import UnfittedContext
from relayspec.vocab_bridge import load_vocab_intersection


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


def select_records(records: list[dict[str, Any]], *, limit: int, seed: int):
    selected = list(records)
    random.Random(seed).shuffle(selected)
    return selected[:limit]


def token_hash(token_ids: torch.Tensor) -> str:
    values = token_ids.detach().to(device="cpu", dtype=torch.int32).contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def run_method(
    name: str,
    generate: Callable[..., Any],
    input_ids: torch.Tensor,
    max_new_tokens: int,
    tokenizer: Any,
) -> dict[str, Any]:
    recorder = CudaRegionRecorder()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    torch.cuda.synchronize()
    started = time.perf_counter()
    stats = generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        return_stats=True,
        profile_recorder=recorder,
    )
    torch.cuda.synchronize()
    request_seconds = time.perf_counter() - started
    regions = recorder.totals(request_seconds)
    generated = stats.output_ids[0, stats.num_input_tokens :]
    time_to_first_token, decode_seconds = generation_timings(
        stats,
        request_seconds=request_seconds,
    )
    target_calls, draft_calls = generation_call_counts(stats)
    row = {
        "method": name,
        "output_hash": token_hash(generated),
        "output_token_ids": generated.detach().to(device="cpu", dtype=torch.int32).tolist(),
        "completion": tokenizer.decode(generated, skip_special_tokens=True),
        "input_tokens": int(stats.num_input_tokens),
        "output_tokens": int(stats.num_output_tokens),
        "time_to_first_token_seconds": time_to_first_token,
        "decode_seconds": decode_seconds,
        "request_seconds": request_seconds,
        "acceptance_length": statistics.fmean(stats.acceptance_lengths),
        "acceptance_lengths": [int(value) for value in stats.acceptance_lengths],
        "target_calls": target_calls,
        "draft_calls": draft_calls,
        "allocated_memory_before_bytes": allocated_before,
        "reserved_memory_before_bytes": reserved_before,
        "allocated_memory_after_bytes": torch.cuda.memory_allocated(),
        "reserved_memory_after_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_memory_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_memory_bytes": torch.cuda.max_memory_reserved(),
        "profile_regions_ms": regions,
    }
    row.update(optional_generation_stats(stats))
    return row


def main() -> None:
    args = parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    probe = payload.pop("relay_probe", {})
    config = namespace(payload)
    source_config = config.source_trunk
    precision = str(getattr(config.benchmark, "precision", "bfloat16"))
    if precision not in {"bfloat16", "float32"}:
        raise ValueError("DFlash benchmark precision must be bfloat16 or float32")
    runtime_dtype = getattr(torch, precision)
    head_precision = str(getattr(config.benchmark, "target_head_precision", precision))
    if head_precision not in {precision, "float32"}:
        raise ValueError("Target head precision must match runtime or be float32")
    if precision == "float32":
        # Numerical diagnostic: use FP32 for every model and mapper, without TF32.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    method_names = tuple(str(value) for value in config.benchmark.methods)
    unload_source_trunk = bool(getattr(config.benchmark, "unload_source_trunk", False))
    variants = probe.get("variants", {})
    model_methods = campaign_model_methods(method_names, variants)
    load_plan = model_load_plan(
        model_methods,
        unload_source_trunk=unload_source_trunk,
    )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    expected_workers = (
        1 if getattr(config.benchmark, "research_single_gpu", False) else 4
    )
    if world_size != expected_workers:
        raise RuntimeError("relay probe requires exactly four GPUs")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
        timeout=timedelta(hours=6),
    )
    device = torch.device(f"cuda:{local_rank}")
    cache_dir = Path(os.environ["TRANSFORMERS_CACHE"])
    draft_class, official_dflash_generate = import_official_dflash(
        os.environ["DFLASH_SOURCE"], config.proposer.source_commit
    )
    source = None
    if load_plan["load_source"]:
        source = (
            AutoModelForCausalLM.from_pretrained(
                source_config.model.id,
                revision=source_config.model.revision,
                cache_dir=cache_dir,
                attn_implementation="sdpa",
                dtype=runtime_dtype,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )
    # Release unused source layers before allocating the target (notably FP32).
    source_provider = None
    if load_plan["build_source_provider"]:
        assert source is not None
        source_provider = (
            SourceTapProvider(
                source,
                source_layers=source_config.layers,
                tap_layers=tuple(source_config.tap_layers),
            )
            .to(device=device, dtype=runtime_dtype)
            .eval()
        )
    source_embedding = None
    source_lm_head = None
    if source is not None:
        source_embedding = source.model.embed_tokens
        source_lm_head = source.lm_head
    if load_plan["unload_source_trunk"]:
        assert source_embedding is not None and source_lm_head is not None
        del source
        source = None
        gc.collect()
        torch.cuda.empty_cache()
    target = (
        AutoModelForCausalLM.from_pretrained(
            config.target.id,
            revision=resolve_revision(config.target.id, config.target.revision),
            cache_dir=cache_dir,
            attn_implementation="sdpa",
            dtype=runtime_dtype,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    head_diagnostic = None
    if precision == "bfloat16" and head_precision == "float32":
        from relayspec.numerical_diagnostics import promote_output_head

        torch.backends.cuda.matmul.allow_tf32 = False
        head_diagnostic = promote_output_head(target)
    draft = (
        draft_class.from_pretrained(
            config.proposer.id,
            revision=config.proposer.revision,
            cache_dir=cache_dir,
            attn_implementation="sdpa",
            dtype=runtime_dtype,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    native_target_draft = None
    if load_plan["load_native_target_draft"]:
        if not hasattr(config, "native_target_proposer"):
            raise ValueError(
                "native_target_dflash requires native_target_proposer config"
            )
        native_target_draft = (
            draft_class.from_pretrained(
                config.native_target_proposer.id,
                revision=config.native_target_proposer.revision,
                cache_dir=cache_dir,
                attn_implementation="sdpa",
                dtype=runtime_dtype,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )
    needs_relay = bool(
        set(method_names) & {"relay_f", "relay_p", "relay_p_cross_family"}
    )
    relay = None
    target_layer_ids: tuple[int, ...] = ()
    if needs_relay:
        checkpoint_path = os.environ.get(
            "RELAYSPEC_CHECKPOINT",
            probe["checkpoint_path"],
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        target_layer_ids = tuple(int(value) for value in checkpoint["target_layer_ids"])
        relay_architecture = str(
            checkpoint.get("relay_architecture", "normalized_linear")
        )
        adapter_input_width = checkpoint.get("adapter_input_width")
        delta_rank = checkpoint.get("delta_rank")
        delta_nonlinear = bool(checkpoint.get("delta_nonlinear", False))
        mlp_hidden_width = checkpoint.get("mlp_hidden_width")
        factorized_rank = checkpoint.get("factorized_rank")
        # E4-P1: an adapted checkpoint's projection weight belongs to the
        # frozen base relay's native target, not the target being served
        # here, so its width must come from the checkpoint, not from this
        # run's target config. The MLP ablation (rigor pass) replaces the
        # single `projection.weight` with a two-layer `nn.Sequential`, whose
        # state dict keys are `projection.0.weight`/`projection.2.weight`
        # instead, so which key to read the input width from depends on
        # which architecture this checkpoint actually is.
        projection_weight_key = (
            "projection.0.weight"
            if mlp_hidden_width is not None or factorized_rank is not None
            else "projection.weight"
        )
        native_input_width = checkpoint["relay"][projection_weight_key].shape[1]
        native_target_hidden_size = native_input_width // len(target_layer_ids)
        relay = TargetFeatureRelay(
            target_hidden_size=native_target_hidden_size,
            num_taps=len(target_layer_ids),
            draft_hidden_size=draft.config.hidden_size,
            eps=target.config.rms_norm_eps,
            normalize_input=relay_architecture == "normalized_linear",
            adapter_input_width=adapter_input_width,
            delta_rank=delta_rank,
            delta_nonlinear=delta_nonlinear,
            mlp_hidden_width=mlp_hidden_width,
            factorized_rank=factorized_rank,
        )
        relay.load_state_dict(checkpoint["relay"], strict=True)
        relay = relay.to(device=device, dtype=runtime_dtype).eval()
    variant_mappers = {}
    variant_provenance = {}
    drafter_updates = probe.get("drafter_updates", {})
    if set(drafter_updates) - set(variants):
        raise ValueError("every adapted drafter requires an explicit mapper variant")
    variant_drafters = {}
    for name, checkpoint_path in variants.items():
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("proposer_family", "dflash") != "dflash":
            raise ValueError(
                "DFlash campaign received a different drafter family's mapper"
            )
        mapper, taps = restore_mapper(
            checkpoint,
            target_hidden_size=target.config.hidden_size,
            draft_hidden_size=draft.config.hidden_size,
            eps=target.config.rms_norm_eps,
        )
        if taps[-1] >= target.config.num_hidden_layers:
            raise ValueError("mapper taps lie outside the target")
        mapper = mapper.to(device=device, dtype=runtime_dtype).eval()
        variant_mappers[name] = (mapper, taps)
        variant_provenance[name] = {
            "checkpoint": str(checkpoint_path),
            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "parameters": sum(p.numel() for p in mapper.parameters()),
            "resident_parameter_bytes": sum(
                p.numel() * p.element_size() for p in mapper.parameters()
            ),
            "target_layer_ids": taps,
        }
        del checkpoint
        if name in drafter_updates:
            adaptation_path = Path(drafter_updates[name])
            adaptation = torch.load(
                adaptation_path, map_location="cpu", weights_only=True
            )
            adapted = copy.deepcopy(draft).requires_grad_(False).eval()
            apply_merged_lora(
                adapted,
                adaptation,
                proposer=payload["proposer"],
                mapper_sha256=variant_provenance[name]["sha256"],
            )
            variant_drafters[name] = adapted
            variant_provenance[name]["drafter_update"] = {
                "checkpoint": str(adaptation_path),
                "sha256": hashlib.sha256(adaptation_path.read_bytes()).hexdigest(),
                "base_sha256": adaptation["base_sha256"],
                "updated_weights": sorted(adaptation["weights"]),
                "resident_parameter_bytes": sum(
                    p.numel() * p.element_size() for p in adapted.parameters()
                ),
            }
            del adaptation
    unfitted = {}
    for name in {"direct_slice", "frozen_fc_slice"} & set(method_names):
        unfitted[name] = (
            UnfittedContext(
                target_width=target.config.hidden_size,
                draft_width=draft.config.hidden_size,
                num_taps=len(source_config.tap_layers),
                source_fc=draft.fc if name == "frozen_fc_slice" else None,
            )
            .to(device=device, dtype=runtime_dtype)
            .eval()
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config.target.id,
        revision=resolve_revision(config.target.id, config.target.revision),
        cache_dir=cache_dir,
        local_files_only=True,
    )
    source_tokenizer = None
    source_to_target_intersection = None
    target_to_source_intersection = None
    if "relay_p_cross_family" in method_names:
        source_tokenizer = AutoTokenizer.from_pretrained(
            source_config.model.id,
            revision=source_config.model.revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        bridge_dir = getattr(config.benchmark, "vocab_bridge_dir", None)
        if bridge_dir is not None:
            source_to_target_name = getattr(
                config.benchmark, "vocab_bridge_source_to_target", None
            )
            target_to_source_name = getattr(
                config.benchmark, "vocab_bridge_target_to_source", None
            )
            if source_to_target_name is None or target_to_source_name is None:
                raise ValueError(
                    "vocab_bridge_dir requires both vocab_bridge_source_to_target "
                    "and vocab_bridge_target_to_source"
                )
            source_to_target_intersection = load_vocab_intersection(
                Path(bridge_dir) / source_to_target_name
            )
            target_to_source_intersection = load_vocab_intersection(
                Path(bridge_dir) / target_to_source_name
            )
    manifest = json.loads(
        Path(config.benchmark.manifest_path).read_text(encoding="utf-8")
    )
    candidates = [
        row
        for row in manifest["records"]
        if row["benchmark"] in config.benchmark.benchmarks
    ]
    records = select_records(
        candidates,
        limit=int(config.benchmark.max_prompts),
        seed=config.seed,
    )
    if len(records) != int(config.benchmark.max_prompts):
        raise RuntimeError(
            f"requested {config.benchmark.max_prompts} prompts but found {len(records)}"
        )
    local_records = shard_records(records, world_size=world_size, rank=rank)
    if not local_records:
        raise RuntimeError(f"rank {rank} received no benchmark prompts")

    common = {
        "stop_token_ids": [tokenizer.eos_token_id],
        "temperature": config.generation.temperature,
        "block_size": config.benchmark.block_size,
    }

    def native(**kwargs: Any) -> Any:
        kwargs.pop("profile_recorder", None)
        return native_autoregressive_generate(
            target,
            stop_token_ids=common["stop_token_ids"],
            temperature=common["temperature"],
            **kwargs,
        )

    def baseline(**kwargs: Any) -> Any:
        if source_provider is None:
            raise RuntimeError("source provider was not loaded")
        return matched_full_target_dflash_generate(
            draft,
            source_provider=source_provider,
            native_target=target,
            **common,
            **kwargs,
        )

    def target_specific(**kwargs: Any) -> Any:
        if native_target_draft is None:
            raise RuntimeError("native_target_proposer is not configured")
        kwargs.pop("profile_recorder", None)
        return official_dflash_generate(
            native_target_draft,
            target=target,
            stop_token_ids=common["stop_token_ids"],
            temperature=common["temperature"],
            block_size=common["block_size"],
            **kwargs,
        )

    def relayed(**kwargs: Any) -> Any:
        if source_embedding is None or source_lm_head is None:
            raise RuntimeError("source embedding/head weights were not loaded")
        return relay_dflash_generate(
            draft,
            relay=relay,
            relay_target_layer_ids=target_layer_ids,
            native_target=target,
            source_embedding=source_embedding,
            source_lm_head=source_lm_head,
            **common,
            **kwargs,
        )

    def relayed_cross_family(**kwargs: Any) -> Any:
        if source_embedding is None or source_lm_head is None:
            raise RuntimeError("source embedding/head weights were not loaded")
        if source_tokenizer is None:
            raise RuntimeError("source tokenizer was not loaded")
        return cross_family_relay_dflash_generate(
            draft,
            relay=relay,
            relay_target_layer_ids=target_layer_ids,
            native_target=target,
            source_embedding=source_embedding,
            source_lm_head=source_lm_head,
            source_tokenizer=source_tokenizer,
            target_tokenizer=tokenizer,
            source_to_target_intersection=source_to_target_intersection,
            target_to_source_intersection=target_to_source_intersection,
            **common,
            **kwargs,
        )

    def unfitted_generate(name, **kwargs):
        return relay_dflash_generate(
            draft,
            relay=unfitted[name],
            relay_target_layer_ids=tuple(source_config.tap_layers),
            native_target=target,
            source_embedding=source_embedding,
            source_lm_head=source_lm_head,
            **common,
            **kwargs,
        )

    available_methods = {
        "direct_slice": lambda **kw: unfitted_generate("direct_slice", **kw),
        "frozen_fc_slice": lambda **kw: unfitted_generate("frozen_fc_slice", **kw),
        "native_ar": native,
        "native_target_dflash": target_specific,
        "naive_source_reuse": baseline,
        "optimized_source_reuse": baseline,
        "relay_f": relayed,
        "relay_p": relayed,
        "relay_p_cross_family": relayed_cross_family,
    }

    def variant_generator(
        mapper,
        taps,
        inherited_draft,
        selective_capture=False,
        release_capture_buffers=False,
        variant_block_size=None,
    ):
        def generate(**kwargs):
            if source_embedding is None or source_lm_head is None:
                raise RuntimeError("mapper campaign requires source embedding/head")
            variant_common = dict(common)
            if variant_block_size is not None:
                variant_common["block_size"] = int(variant_block_size)
            if bool(probe.get("cross_family_variants", False)):
                if source_tokenizer is None:
                    raise RuntimeError("cross-family variants require the cross-family reference method")
                if selective_capture or release_capture_buffers:
                    raise ValueError("selective capture is not supported for cross-family variants")
                return cross_family_relay_dflash_generate(
                    inherited_draft, relay=mapper, relay_target_layer_ids=taps,
                    native_target=target, source_embedding=source_embedding,
                    source_lm_head=source_lm_head, source_tokenizer=source_tokenizer,
                    target_tokenizer=tokenizer,
                    source_to_target_intersection=source_to_target_intersection,
                    target_to_source_intersection=target_to_source_intersection,
                    **variant_common, **kwargs,
                )
            return relay_dflash_generate(
                inherited_draft,
                relay=mapper,
                relay_target_layer_ids=taps,
                native_target=target,
                source_embedding=source_embedding,
                source_lm_head=source_lm_head,
                selective_capture=selective_capture,
                release_capture_buffers=release_capture_buffers,
                **variant_common,
                **kwargs,
            )

        return generate

    for name, (mapper, taps) in variant_mappers.items():
        available_methods[name] = variant_generator(
            mapper,
            taps,
            variant_drafters.get(name, draft),
            bool(probe.get("selective_capture", {}).get(name, False)),
            bool(probe.get("release_capture_buffers", {}).get(name, False)),
            probe.get("variant_block_sizes", {}).get(name),
        )
    unknown = sorted(set(method_names) - set(available_methods))
    if unknown:
        raise ValueError(f"unsupported benchmark methods: {unknown}")

    warmup_record = local_records[0]
    warmup_encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": benchmark_turns(warmup_record)[0]}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    warmup_ids = (
        warmup_encoded["input_ids"]
        if hasattr(warmup_encoded, "keys")
        else warmup_encoded
    ).to(device)
    for _ in range(config.benchmark.warmups):
        for method_name in method_names:
            available_methods[method_name](
                input_ids=warmup_ids,
                max_new_tokens=16,
                return_stats=True,
            )
    repetitions = int(probe.get("repetitions", 1))
    if repetitions <= 0:
        raise ValueError("relay probe repetitions must be positive")
    output_dir = Path(os.environ["RELAYSPEC_OUTPUT"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if variants and rank == 0:
        (output_dir / "mapper-campaign.json").write_text(
            json.dumps(
                {
                    "variants": variant_provenance,
                    "adapted_drafter_variants": sorted(variant_drafters),
                    "methods": method_names,
                    "selective_capture": probe.get("selective_capture", {}),
                    "release_capture_buffers": probe.get("release_capture_buffers", {}),
                    "cross_family_variants": bool(probe.get("cross_family_variants", False)),
                    "variant_block_sizes": probe.get("variant_block_sizes", {}),
                    "runtime_precision": precision,
                    "target_head_precision": head_precision,
                    "target_head_diagnostic": head_diagnostic,
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "target_parameter_dtypes": sorted(
                        {str(p.dtype) for p in target.parameters()}
                    ),
                    "drafter_parameter_dtypes": sorted(
                        {str(p.dtype) for p in draft.parameters()}
                    ),
                    "mapper_parameter_dtypes": {
                        name: sorted({str(p.dtype) for p in mapper.parameters()})
                        for name, (mapper, _) in variant_mappers.items()
                    },
                    "measurement": "All candidates and baselines measured on the same requests with rotated method order. All candidate maps resident during every arm. Memory is campaign residency, not isolated deployment.",
                },
                indent=2,
            )
            + "\n"
        )
    rank_path = output_dir / f"benchmark-rank{rank}.jsonl"
    rows: list[dict[str, Any]] = []
    with rank_path.open("x", encoding="utf-8") as stream:
        for local_index, record in enumerate(local_records):
            turns = benchmark_turns(record)
            for repetition in range(repetitions):
                histories: dict[str, list[dict[str, str]]] = {
                    name: [] for name in method_names
                }
                for turn_index, user_content in enumerate(turns):
                    order_index = (
                        rank + local_index * world_size + repetition + turn_index
                    )
                    for name in rotate_methods(method_names, order_index):
                        messages = [
                            *histories[name],
                            {"role": "user", "content": user_content},
                        ]
                        if "input_ids" in record:
                            exact_ids = validate_exact_input(
                                record,
                                vocab_size=target.config.vocab_size,
                                max_positions=target.config.max_position_embeddings,
                                output_cap=int(config.generation.max_new_tokens),
                            )
                            input_ids = torch.tensor(
                                [exact_ids], dtype=torch.long, device=device
                            )
                        else:
                            encoded = tokenizer.apply_chat_template(
                                messages,
                                tokenize=True,
                                add_generation_prompt=True,
                                enable_thinking=False,
                                return_tensors="pt",
                            )
                            input_ids = (
                                encoded["input_ids"]
                                if hasattr(encoded, "keys")
                                else encoded
                            ).to(device)
                        sampling_seed = initialize_sampling_rng(
                            float(config.generation.temperature),
                            getattr(config.benchmark, "sampling_seed_base", None),
                            record["problem_id"],
                            repetition,
                            turn_index,
                        )
                        row = run_method(
                            name,
                            available_methods[name],
                            input_ids,
                            config.generation.max_new_tokens,
                            tokenizer,
                        )
                        if sampling_seed is not None:
                            row["sampling_seed"] = sampling_seed
                            row["sampling_temperature"] = float(
                                config.generation.temperature
                            )
                        if name in variant_provenance:
                            row["mapper_checkpoint_sha256"] = variant_provenance[name][
                                "sha256"
                            ]
                            if name in variant_drafters:
                                row["drafter_checkpoint_sha256"] = variant_provenance[
                                    name
                                ]["drafter_update"]["sha256"]
                        histories[name] = [
                            *messages,
                            {"role": "assistant", "content": row["completion"]},
                        ]
                        problem_id = str(record["problem_id"])
                        if len(turns) > 1:
                            problem_id = f"{problem_id}/turn{turn_index}"
                        row.update(
                            {
                                "benchmark": record["benchmark"],
                                "problem_id": problem_id,
                                "reference_answer": record.get("answer"),
                                "rank": rank,
                                "repetition": repetition,
                                "turn_index": turn_index,
                            }
                        )
                        rows.append(row)
                        serialized = json.dumps(row, sort_keys=True)
                        stream.write(serialized + "\n")
                        stream.flush()
                        print(
                            json.dumps(
                                {"event": "benchmark_row", **row}, sort_keys=True
                            ),
                            flush=True,
                        )
    dist.barrier()
    if rank == 0:
        merged: list[dict[str, Any]] = []
        for worker in range(world_size):
            with (output_dir / f"benchmark-rank{worker}.jsonl").open(
                encoding="utf-8"
            ) as worker_stream:
                merged.extend(
                    json.loads(line) for line in worker_stream if line.strip()
                )
        summary = aggregate_rows(
            merged,
            methods=method_names,
            native_method=method_names[0],
        )
        source_method = next(
            (
                name
                for name in ("optimized_source_reuse", "naive_source_reuse")
                if name in summary["methods"]
            ),
            None,
        )
        if source_method is not None and "relay_p" in summary["methods"]:
            summary["methods"]["relay_p"]["speedup_vs_naive_source_reuse"] = (
                summary["methods"]["relay_p"]["decode_tokens_per_second"]
                / summary["methods"][source_method]["decode_tokens_per_second"]
            )
        (output_dir / "benchmark-summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"event": "benchmark_summary", **summary}, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
