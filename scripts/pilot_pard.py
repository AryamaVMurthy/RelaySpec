"""Four-GPU correctness/resource pilot for released PARD and matched eager AR."""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download

from relayspec.benchmarking import benchmark_turns
from relayspec.pard_adapter import (
    PARD_SOURCE_SHA256,
    load_instrumented_pard,
    trim_pard_tokens,
)


@torch.no_grad()
def eager_ar(model, cache, ids, cap, eos):
    torch.cuda.synchronize()
    started = time.perf_counter()
    cache.reset()
    current = ids
    position = 0
    outputs = []
    for _ in range(cap):
        output = model(
            input_ids=current,
            position_ids=None,
            past_key_values=cache,
            cache_position=torch.arange(
                position, position + current.shape[1], device=ids.device
            ),
            use_cache=True,
            attention_mask=None,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        position += current.shape[1]
        current = output.logits[:, -1:].argmax(-1)
        outputs.append(current)
        if int(current.item()) == eos:
            break
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return torch.cat(outputs, dim=1)[0].cpu().tolist(), seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    setup_gate = json.loads(Path(os.environ["PARD_SETUP_GATE"]).read_text())
    if (
        setup_gate["status"] != "pass"
        or setup_gate["config_sha256"]
        != hashlib.sha256(args.config.read_bytes()).hexdigest()
        or config["pilot"]
        != {
            "requests": 4,
            "max_new_tokens": 128,
            "draft_k": 12,
            "max_cache_len": 2048,
            "compiled": False,
            "attention": "eager",
            "dtype": "bfloat16",
            "warmup_tokens": 16,
        }
    ):
        raise ValueError("pilot differs from its declared settings or verified setup")
    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4 or torch.cuda.device_count() != 4:
        raise ValueError("PARD pilot requires four independent GPU workers")
    torch.cuda.set_device(rank)
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    source = Path(os.environ["PARD_SOURCE"]) / "pard/pard_infer.py"
    pard_class = load_instrumented_pard(source)
    paths = {
        key: snapshot_download(
            repo_id=config[key]["id"],
            revision=config[key]["revision"],
            cache_dir=os.environ["TRANSFORMERS_CACHE"],
            local_files_only=True,
        )
        for key in ("target", "proposer")
    }
    infer = pard_class(
        draft_k=12,
        tokens=128,
        draft=paths["proposer"],
        target=paths["target"],
        benchmark="math500",
        para=True,
        nc=True,
        max_cache_len=2048,
        model_serie="qwen",
    )
    infer.log = False
    setup_started = time.perf_counter()
    infer.get_model(
        infer.checkpoint_draft,
        infer.checkpoint_target,
        infer.model_serie,
        infer.torch_dtype,
        infer.max_cache_len,
        True,
        False,
    )
    infer.model.eval().requires_grad_(False)
    infer.model_draft.eval().requires_grad_(False)
    torch.manual_seed(config["seed"])
    records = [
        r
        for r in json.loads(Path(config["manifest_path"]).read_text())["records"]
        if r["benchmark"] == "math500"
    ]
    random.Random(config["seed"]).shuffle(records)
    if len(records) < 4:
        raise ValueError("missing PARD pilot requests")
    record = records[rank]
    encoded = infer.tokenizer.apply_chat_template(
        [{"role": "user", "content": benchmark_turns(record)[0]}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    ids = (encoded["input_ids"] if hasattr(encoded, "keys") else encoded).to(
        f"cuda:{rank}"
    )
    eos = infer.tokenizer.eos_token_id
    # Same first16-token warmup for both paths, always excluded from observations.
    eager_ar(infer.model, infer.target_cache, ids, 16, eos)
    infer.tokens = 16
    infer.set_input_ids(ids)
    infer.generate(
        ["pretokenized"],
        infer.model,
        infer.model_draft,
        infer.target_cache,
        infer.draft_cache,
        infer.tokenizer,
    )
    infer.tokens = 128
    setup_seconds = time.perf_counter() - setup_started
    rows = {}
    names = ["eager_ar", "pard", "pard_duplicate"]
    names = names[rank % 3 :] + names[: rank % 3]
    for name in names:
        torch.cuda.reset_peak_memory_stats()
        if name == "eager_ar":
            tokens, seconds = eager_ar(infer.model, infer.target_cache, ids, 128, eos)
            details = {
                "target_calls": len(tokens),
                "draft_calls": 0,
                "raw_output_tokens": len(tokens),
            }
        else:
            infer.set_input_ids(ids)
            infer.generate(
                ["pretokenized"],
                infer.model,
                infer.model_draft,
                infer.target_cache,
                infer.draft_cache,
                infer.tokenizer,
            )
            captured = infer.captured_request
            raw = captured["token_ids"][0].cpu().tolist()
            tokens = trim_pard_tokens(raw, max_new_tokens=128, eos_token_id=eos)
            seconds = captured["request_seconds"]
            details = {
                k: captured[k]
                for k in ("accepted_lengths", "target_calls", "draft_calls")
            }
            details["raw_output_tokens"] = len(raw)
            details["raw_token_ids"] = raw
        rows[name] = {
            "method": name,
            "rank": rank,
            "problem_id": record["problem_id"],
            "token_ids": tokens,
            "output_tokens": len(tokens),
            "request_seconds": seconds,
            "input_tokens": ids.shape[1],
            "input_ids": ids[0].cpu().tolist(),
            "completion": infer.tokenizer.decode(tokens, skip_special_tokens=True),
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            **details,
        }
    path = output / f"pard-rank{rank}.json"
    path.write_text(
        json.dumps({"rows": rows, "setup_seconds": setup_seconds}, indent=2) + "\n"
    )
    # Fail without hiding raw output when deterministic correctness differs.
    if not (
        rows["eager_ar"]["token_ids"]
        == rows["pard"]["token_ids"]
        == rows["pard_duplicate"]["token_ids"]
    ):
        raise ValueError("PARD pilot differs from matched eager AR or its duplicate")
    if any(
        rows["pard"][field] != rows["pard_duplicate"][field]
        for field in (
            "accepted_lengths",
            "raw_token_ids",
            "target_calls",
            "draft_calls",
        )
    ):
        raise ValueError("duplicate PARD proposal trajectory differs")
    (output / f"pard-gate-rank{rank}.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "rank": rank,
                "exact_ar_and_duplicate_outputs": True,
                "rows_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "upstream_source_sha256": PARD_SOURCE_SHA256,
                "scope": "Released PARD parallel algorithm, uncompiled eager attention and static caches. "
                "Matched eager AR on the identical pretokenized user prompt. Request latency "
                "includes prefills/cache reset and excludes detokenization. Upstream aggregate "
                "TPS is unused. Raw cap/EOS overshoot preserved; counted output is trimmed "
                "without subtracting computation time. Resource/correctness pilot only. "
                "No cross-engine RelaySpec speed ranking or full-answer quality claim.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
