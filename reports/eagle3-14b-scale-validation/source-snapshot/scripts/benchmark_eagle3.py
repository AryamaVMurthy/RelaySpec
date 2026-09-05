from __future__ import annotations

import argparse
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
    generation_call_counts,
    generation_timings,
    optional_generation_stats,
    rotate_methods,
    shard_records,
)
from relayspec.eagle3 import eagle3_generate, import_official_eagle3
from relayspec.generation import native_autoregressive_generate
from relayspec.profiling import CudaRegionRecorder
from relayspec.proposers import RelayContextProvider, SourceContextProvider
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

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
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
    if "native_target_eagle3" in methods:
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
        )
        relay.load_state_dict(checkpoint["relay"], strict=True)
        relay = relay.to(device=device, dtype=torch.bfloat16).eval()

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
        if relay is None:
            raise RuntimeError("relay was not loaded")
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
    unknown = sorted(set(methods) - set(available))
    if unknown:
        raise ValueError(f"unsupported EAGLE-3 methods: {unknown}")

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
    rank_path = output_dir / f"benchmark-rank{rank}.jsonl"
    rows: list[dict[str, Any]] = []
    with rank_path.open("x", encoding="utf-8") as stream:
        for local_index, record in enumerate(local_records):
            for repetition in range(repetitions):
                for method in rotate_methods(
                    methods, rank + local_index * world_size + repetition
                ):
                    prompt = benchmark_turns(record)[0]
                    encoded = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=False,
                        return_tensors="pt",
                    )
                    input_ids = (
                        encoded["input_ids"] if hasattr(encoded, "keys") else encoded
                    ).to(device)
                    row = run_method(
                        method,
                        available[method],
                        input_ids,
                        int(config.generation.max_new_tokens),
                        tokenizer,
                    )
                    row.update(
                        {
                            "benchmark": record["benchmark"],
                            "problem_id": str(record["problem_id"]),
                            "reference_answer": record.get("answer"),
                            "rank": rank,
                            "repetition": repetition,
                            "turn_index": 0,
                        }
                    )
                    rows.append(row)
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    print(
                        json.dumps({"event": "benchmark_row", **row}, sort_keys=True),
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
