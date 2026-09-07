from __future__ import annotations

import argparse
import copy
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
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from relayspec.benchmarking import (
    aggregate_rows,
    benchmark_turns,
    eagle3_load_plan,
    generation_call_counts,
    generation_timings,
    optional_generation_stats,
    rotate_methods,
    shard_records,
)
from relayspec.drafter_adaptation import apply_merged_lora
from relayspec.eagle3 import eagle3_generate, import_official_eagle3
from relayspec.generation import native_autoregressive_generate
from relayspec.mapper_campaign import campaign_model_methods, restore_mapper
from relayspec.profiling import CudaRegionRecorder
from relayspec.proposers import RelayContextProvider, SourceContextProvider
from relayspec.relay import TargetFeatureRelay
from relayspec.sampling_rng import initialize_sampling_rng
from relayspec.sequence_scaling import validate_exact_input
from relayspec.source import SourceTapProvider


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


def select_records(
    records: list[dict[str, Any]], *, limit: int, seed: int
) -> list[dict[str, Any]]:
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
        profile_recorder=recorder,
    )
    torch.cuda.synchronize()
    request_seconds = time.perf_counter() - started
    generated = stats.output_ids[0, stats.num_input_tokens :]
    ttft, decode_seconds = generation_timings(stats, request_seconds=request_seconds)
    target_calls, draft_calls = generation_call_counts(stats)
    row = {
        "method": name,
        "output_hash": token_hash(generated),
        "completion": tokenizer.decode(generated, skip_special_tokens=True),
        "input_tokens": int(stats.num_input_tokens),
        "output_tokens": int(stats.num_output_tokens),
        "time_to_first_token_seconds": ttft,
        "decode_seconds": decode_seconds,
        "request_seconds": request_seconds,
        "acceptance_length": (
            statistics.fmean(stats.acceptance_lengths)
            if stats.acceptance_lengths
            else 0.0
        ),
        "acceptance_lengths": [int(value) for value in stats.acceptance_lengths],
        "target_calls": target_calls,
        "draft_calls": draft_calls,
        "allocated_memory_before_bytes": allocated_before,
        "reserved_memory_before_bytes": reserved_before,
        "allocated_memory_after_bytes": torch.cuda.memory_allocated(),
        "reserved_memory_after_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_memory_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_memory_bytes": torch.cuda.max_memory_reserved(),
        "profile_regions_ms": recorder.totals(request_seconds),
    }
    row.update(optional_generation_stats(stats))
    return row


def main() -> None:
    args = parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    probe = payload.pop("relay_probe", {})
    config = namespace(payload)
    methods = tuple(str(value) for value in config.benchmark.methods)
    variants = probe.get("variants", {})
    model_methods = campaign_model_methods(
        methods, variants, relay_method="relay_eagle3"
    )
    supported_methods = {
        "native_ar",
        "source_reuse_eagle3",
        "native_target_eagle3",
        "relay_eagle3",
    }
    unknown = sorted(set(model_methods) - supported_methods)
    if unknown:
        raise ValueError(f"unsupported EAGLE-3 methods: {unknown}")
    load_plan = eagle3_load_plan(model_methods)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    expected_workers = (
        1 if getattr(config.benchmark, "research_single_gpu", False) else 4
    )
    if world_size != expected_workers:
        raise RuntimeError("EAGLE-3 benchmark requires exactly four GPUs")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
        timeout=timedelta(hours=6),
    )
    device = torch.device(f"cuda:{local_rank}")
    cache_dir = Path(os.environ["TRANSFORMERS_CACHE"])
    api = import_official_eagle3(
        os.environ["DEEPSPEC_SOURCE"], config.proposer.source_commit
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
    source = None
    if load_plan["load_source"]:
        source = (
            AutoModelForCausalLM.from_pretrained(
                config.source_trunk.model.id,
                revision=config.source_trunk.model.revision,
                cache_dir=cache_dir,
                attn_implementation="sdpa",
                dtype=torch.bfloat16,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )

    source_draft = None
    if load_plan["load_source_draft"]:
        source_draft = (
            api.model_class.from_pretrained(
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
        source_draft.target_layer_ids = [int(x) for x in source_draft.target_layer_ids]

    source_taps = None
    if source is not None:
        if source_draft is None:
            raise RuntimeError("source reuse requires the source EAGLE-3 proposer")
        expected_taps = tuple(int(x) for x in source_draft.target_layer_ids)
        configured_taps = tuple(int(x) for x in config.source_trunk.tap_layers)
        if expected_taps != configured_taps:
            raise RuntimeError(
                f"source taps do not match released EAGLE-3 checkpoint: "
                f"{configured_taps} != {expected_taps}"
            )
        source_taps = (
            SourceTapProvider(
                source,
                source_layers=int(config.source_trunk.layers),
                tap_layers=configured_taps,
            )
            .to(device=device, dtype=torch.bfloat16)
            .eval()
        )

    target_draft = None
    if load_plan["load_target_draft"]:
        target_draft = (
            api.model_class.from_pretrained(
                config.native_target_proposer.id,
                revision=config.native_target_proposer.revision,
                cache_dir=cache_dir,
                attn_implementation="sdpa",
                dtype=torch.bfloat16,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )
        target_draft.target_layer_ids = [int(x) for x in target_draft.target_layer_ids]

    relay = None
    relay_layer_ids: tuple[int, ...] = ()
    if "relay_eagle3" in methods:
        if source_draft is None:
            raise RuntimeError("relay EAGLE-3 requires the source proposer")
        checkpoint_path = os.environ.get(
            "RELAYSPEC_CHECKPOINT", probe.get("checkpoint_path")
        )
        if not checkpoint_path:
            raise ValueError("relay_eagle3 requires relay_probe.checkpoint_path")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        relay_layer_ids = tuple(int(x) for x in checkpoint["target_layer_ids"])
        relay_architecture = str(
            checkpoint.get("relay_architecture", "normalized_linear")
        )
        relay = TargetFeatureRelay(
            target_hidden_size=target.config.hidden_size,
            num_taps=len(relay_layer_ids),
            draft_hidden_size=source_draft.config.hidden_size,
            eps=target.config.rms_norm_eps,
            normalize_input=relay_architecture == "normalized_linear",
            mlp_hidden_width=checkpoint.get("mlp_hidden_width"),
            factorized_rank=checkpoint.get("factorized_rank"),
        )
        relay.load_state_dict(checkpoint["relay"], strict=True)
        relay = relay.to(device=device, dtype=torch.bfloat16).eval()

    variant_mappers = {}
    variant_provenance = {}
    drafter_updates = probe.get("drafter_updates", {})
    if set(drafter_updates) - set(variants):
        raise ValueError(
            "every adapted EAGLE drafter requires an explicit mapper variant"
        )
    variant_drafters = {}
    for name, checkpoint_path in variants.items():
        if source_draft is None:
            raise ValueError("EAGLE-3 mapper campaign requires its source proposer")
        path = Path(checkpoint_path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("proposer_family") != "eagle3":
            raise ValueError("EAGLE-3 campaign checkpoint has the wrong family")
        mapper, taps = restore_mapper(
            checkpoint,
            target_hidden_size=target.config.hidden_size,
            draft_hidden_size=source_draft.config.hidden_size,
            eps=target.config.rms_norm_eps,
        )
        if taps[-1] >= target.config.num_hidden_layers:
            raise ValueError("mapper taps lie outside the target")
        variant_mappers[name] = (
            mapper.to(device=device, dtype=torch.bfloat16).eval(),
            taps,
        )
        variant_provenance[name] = {
            "checkpoint_path": str(path),
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "parameters": sum(p.numel() for p in mapper.parameters()),
            "target_layer_ids": list(taps),
        }
        del checkpoint
        if name in drafter_updates:
            adaptation_path = Path(drafter_updates[name])
            adaptation = torch.load(
                adaptation_path, map_location="cpu", weights_only=True
            )
            adapted = copy.deepcopy(source_draft).requires_grad_(False).eval()
            apply_merged_lora(
                adapted,
                adaptation,
                proposer=payload["proposer"],
                mapper_sha256=variant_provenance[name]["checkpoint_sha256"],
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

    tokenizer = AutoTokenizer.from_pretrained(
        config.target.id,
        revision=config.target.revision,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    stop_token_ids = [int(tokenizer.eos_token_id)]

    def native(**kwargs: Any) -> Any:
        kwargs.pop("profile_recorder", None)
        return native_autoregressive_generate(
            target,
            stop_token_ids=stop_token_ids,
            temperature=float(config.generation.temperature),
            return_stats=True,
            **kwargs,
        )

    def source_reuse(**kwargs: Any) -> Any:
        if source_taps is None or source_draft is None:
            raise RuntimeError("source-reuse EAGLE-3 components were not loaded")
        recorder = kwargs.pop("profile_recorder", None)
        provider = SourceContextProvider(
            tap_provider=source_taps,
            source_cache=DynamicCache(),
            project=source_draft.project_hidden_states,
            profile_recorder=recorder,
        )
        return eagle3_generate(
            api=api,
            target_model=target,
            draft_model=source_draft,
            context_provider=provider,
            temperature=float(config.generation.temperature),
            stop_token_ids=stop_token_ids,
            max_proposal_tokens=int(config.benchmark.draft_length),
            profile_recorder=recorder,
            **kwargs,
        )

    def native_target(**kwargs: Any) -> Any:
        if target_draft is None:
            raise RuntimeError("native target EAGLE-3 was not loaded")
        recorder = kwargs.pop("profile_recorder", None)
        provider = RelayContextProvider(
            relay=torch.nn.Identity(),
            target_layer_ids=tuple(int(x) for x in target_draft.target_layer_ids),
            postprocess=target_draft.project_hidden_states,
            profile_recorder=recorder,
        )
        return eagle3_generate(
            api=api,
            target_model=target,
            draft_model=target_draft,
            context_provider=provider,
            temperature=float(config.generation.temperature),
            stop_token_ids=stop_token_ids,
            max_proposal_tokens=int(config.benchmark.draft_length),
            profile_recorder=recorder,
            **kwargs,
        )

    def relayed(**kwargs: Any) -> Any:
        if relay is None or source_draft is None:
            raise RuntimeError("relay EAGLE-3 components were not loaded")
        recorder = kwargs.pop("profile_recorder", None)
        provider = RelayContextProvider(
            relay=relay,
            target_layer_ids=relay_layer_ids,
            profile_recorder=recorder,
        )
        return eagle3_generate(
            api=api,
            target_model=target,
            draft_model=source_draft,
            context_provider=provider,
            temperature=float(config.generation.temperature),
            stop_token_ids=stop_token_ids,
            max_proposal_tokens=int(config.benchmark.draft_length),
            profile_recorder=recorder,
            **kwargs,
        )

    available = {
        "native_ar": native,
        "source_reuse_eagle3": source_reuse,
        "native_target_eagle3": native_target,
        "relay_eagle3": relayed,
    }

    def variant_generator(mapper, taps, selected_draft):
        def generate(**kwargs):
            recorder = kwargs.pop("profile_recorder", None)
            provider = RelayContextProvider(
                relay=mapper,
                target_layer_ids=taps,
                profile_recorder=recorder,
            )
            return eagle3_generate(
                api=api,
                target_model=target,
                draft_model=selected_draft,
                context_provider=provider,
                temperature=float(config.generation.temperature),
                stop_token_ids=stop_token_ids,
                max_proposal_tokens=int(config.benchmark.draft_length),
                profile_recorder=recorder,
                **kwargs,
            )

        return generate

    for name, (mapper, taps) in variant_mappers.items():
        available[name] = variant_generator(
            mapper, taps, variant_drafters.get(name, source_draft)
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
        seed=int(config.seed),
    )
    if len(records) != int(config.benchmark.max_prompts):
        raise RuntimeError("manifest contains fewer prompts than requested")
    local_records = shard_records(records, world_size=world_size, rank=rank)

    warmup = tokenizer.apply_chat_template(
        [{"role": "user", "content": benchmark_turns(local_records[0])[0]}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    warmup_ids = (warmup["input_ids"] if hasattr(warmup, "keys") else warmup).to(device)
    for _ in range(int(config.benchmark.warmups)):
        for method in methods:
            available[method](input_ids=warmup_ids, max_new_tokens=16)

    repetitions = int(probe.get("repetitions", 1))
    output_dir = Path(os.environ["RELAYSPEC_OUTPUT"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if variants and rank == 0:
        (output_dir / "mapper-campaign.json").write_text(
            json.dumps(
                {
                    "variants": variant_provenance,
                    "scope": "Paired EAGLE-3 candidate decoding with shared frozen models, fresh request providers and rotated method order. Resident memory includes all candidates; it is not isolated deployment memory.",
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
                    method: [] for method in methods
                }
                for turn_index, user_content in enumerate(turns):
                    order_index = (
                        rank + local_index * world_size + repetition + turn_index
                    )
                    for method in rotate_methods(methods, order_index):
                        messages = [
                            *histories[method],
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
                            method,
                            available[method],
                            input_ids,
                            int(config.generation.max_new_tokens),
                            tokenizer,
                        )
                        if sampling_seed is not None:
                            row["sampling_seed"] = sampling_seed
                            row["sampling_temperature"] = float(
                                config.generation.temperature
                            )
                        if method in variant_provenance:
                            row["mapper_checkpoint_sha256"] = variant_provenance[
                                method
                            ]["checkpoint_sha256"]
                            if "drafter_update" in variant_provenance[method]:
                                row["drafter_update_sha256"] = variant_provenance[
                                    method
                                ]["drafter_update"]["sha256"]
                        histories[method] = [
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
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
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
            path = output_dir / f"benchmark-rank{worker}.jsonl"
            merged.extend(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            )
        summary = aggregate_rows(merged, methods=methods, native_method=methods[0])
        (output_dir / "benchmark-summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"event": "benchmark_summary", **summary}, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
