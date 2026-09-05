from __future__ import annotations

import argparse
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
    rotate_methods,
    shard_records,
)
from relayspec.dflash import import_official_dflash
from relayspec.generation import (
    matched_full_target_dflash_generate,
    native_autoregressive_generate,
    relay_dflash_generate,
)
from relayspec.profiling import CudaRegionRecorder
from relayspec.relay import TargetFeatureRelay
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
    probe = payload.pop("relay_probe")
    config = namespace(payload)
    source_config = config.source_trunk
    method_names = tuple(str(value) for value in config.benchmark.methods)
    unload_source_trunk = bool(getattr(config.benchmark, "unload_source_trunk", False))
    load_plan = model_load_plan(
        method_names,
        unload_source_trunk=unload_source_trunk,
    )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
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
                dtype=torch.bfloat16,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )
    source_provider = None
    if load_plan["build_source_provider"]:
        assert source is not None
        source_provider = (
            SourceTapProvider(
                source,
                source_layers=source_config.layers,
                tap_layers=tuple(source_config.tap_layers),
            )
            .to(device=device, dtype=torch.bfloat16)
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
    checkpoint_path = os.environ.get(
        "RELAYSPEC_CHECKPOINT",
        probe["checkpoint_path"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    target_layer_ids = tuple(int(value) for value in checkpoint["target_layer_ids"])
    relay_architecture = str(checkpoint.get("relay_architecture", "normalized_linear"))
    relay = TargetFeatureRelay(
        target_hidden_size=target.config.hidden_size,
        num_taps=len(target_layer_ids),
        draft_hidden_size=draft.config.hidden_size,
        eps=target.config.rms_norm_eps,
        normalize_input=relay_architecture == "normalized_linear",
    )
    relay.load_state_dict(checkpoint["relay"], strict=True)
    relay = relay.to(device=device, dtype=torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        config.target.id,
        revision=config.target.revision,
        cache_dir=cache_dir,
        local_files_only=True,
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

    available_methods = {
        "native_ar": native,
        "native_target_dflash": target_specific,
        "naive_source_reuse": baseline,
        "optimized_source_reuse": baseline,
        "relay_f": relayed,
        "relay_p": relayed,
    }
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
                        row = run_method(
                            name,
                            available_methods[name],
                            input_ids,
                            config.generation.max_new_tokens,
                            tokenizer,
                        )
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
