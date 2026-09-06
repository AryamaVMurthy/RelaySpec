"""Prospective PARD comparison with a verifier-observed replica of every request."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from pilot_pard import eager_ar

from relayspec.benchmarking import benchmark_turns
from relayspec.pard_adapter import (
    load_instrumented_pard,
    trim_pard_tokens,
    verify_pard_decisions,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source_path = Path(config["source_config"])
    source_config = yaml.safe_load(source_path.read_text())
    protocol_path = Path(config["verification_protocol"])
    protocol = json.loads(protocol_path.read_text())
    setup_path = Path(os.environ["PARD_SETUP_GATE"])
    setup = json.loads(setup_path.read_text())
    for path, sha in config.get("prerequisites", {}).items():
        prerequisite = Path(path)
        if (
            digest(prerequisite) != sha
            or json.loads(prerequisite.read_text())["status"] != "complete"
        ):
            raise ValueError("PARD longer-output prerequisite did not pass")
    if config.get("phase") in {"quality_pilot", "quality_full"}:
        expected_requests = protocol["development"][
            "quality_pilot_requests"
            if config["phase"] == "quality_pilot"
            else "quality_requests"
        ]
        if (
            config["requests"] != expected_requests
            or config["max_new_tokens"] != protocol["development"]["quality_token_cap"]
            or not config.get("prerequisites")
        ):
            raise ValueError("PARD quality campaign differs from prospective protocol")
    if (
        digest(source_path) != config["source_config_sha256"]
        or digest(protocol_path) != config["verification_protocol_sha256"]
        or any(digest(Path(p)) != sha for p, sha in protocol["input_sha256"].items())
        or setup["status"] != "pass"
        or setup["config_sha256"] != digest(source_path)
    ):
        raise ValueError("PARD source, setup or prospective protocol changed")
    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4 or torch.cuda.device_count() != 4:
        raise ValueError("PARD campaign requires four GPU workers")
    torch.cuda.set_device(rank)
    torch.manual_seed(config["seed"])
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    source = Path(os.environ["PARD_SOURCE"]) / "pard/pard_infer.py"
    observed_class = load_instrumented_pard(source, trace_decisions=True)
    plain_class = load_instrumented_pard(source)
    paths = {
        key: snapshot_download(
            repo_id=source_config[key]["id"],
            revision=source_config[key]["revision"],
            cache_dir=os.environ["TRANSFORMERS_CACHE"],
            local_files_only=True,
        )
        for key in ("target", "proposer")
    }
    infer = observed_class(
        draft_k=config["draft_k"],
        tokens=config["max_new_tokens"],
        draft=paths["proposer"],
        target=paths["target"],
        benchmark="math500",
        para=True,
        nc=True,
        max_cache_len=config["max_cache_len"],
        model_serie="qwen",
    )
    infer.log = False
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
    frozen = [
        (p, p._version, p.data_ptr())
        for m in (infer.model, infer.model_draft)
        for p in m.parameters()
    ]
    if any(p.dtype != torch.bfloat16 for p, _, _ in frozen):
        raise ValueError("PARD model storage differs from public BF16 runtime")
    plain_generate = plain_class.generate.__get__(infer, type(infer))
    observed_generate = infer.generate
    eos = infer.tokenizer.eos_token_id
    records = [
        r
        for r in json.loads(Path(config["manifest_path"]).read_text())["records"]
        if r["benchmark"] == "math500"
    ]
    random.Random(config["seed"]).shuffle(records)
    records = records[: config["requests"]][rank::4]
    if len(records) != config["requests"] // 4:
        raise ValueError("PARD campaign lacks declared requests")

    def generate(ids, cap, observed):
        infer.tokens = cap
        infer.set_input_ids(ids)
        (observed_generate if observed else plain_generate)(
            ["pretokenized"],
            infer.model,
            infer.model_draft,
            infer.target_cache,
            infer.draft_cache,
            infer.tokenizer,
        )
        capture = dict(infer.captured_request)
        capture["token_ids"] = capture["token_ids"][0].cpu().tolist()
        if observed:
            verify_pard_decisions(
                capture["token_ids"],
                capture["accepted_lengths"],
                capture["target_trace"],
            )
        return capture

    with (output / f"benchmark-rank{rank}.jsonl").open("w") as stream:
        for index, record in enumerate(records):
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
            order = config["methods"].copy()
            random.Random(config["seed"] + rank * 4 + index).shuffle(order)
            for name in order:
                if name == "native_ar":
                    eager_ar(
                        infer.model,
                        infer.target_cache,
                        ids,
                        config["warmup_tokens"],
                        eos,
                    )
                    tokens, seconds = eager_ar(
                        infer.model,
                        infer.target_cache,
                        ids,
                        config["max_new_tokens"],
                        eos,
                    )
                    raw, accepted, proof = tokens, [], None
                    target_calls, draft_calls = len(tokens), 0
                else:
                    generate(ids, config["warmup_tokens"], False)
                    timed = generate(ids, config["max_new_tokens"], False)
                    proof = generate(ids, config["max_new_tokens"], True)
                    if any(
                        timed[k] != proof[k]
                        for k in (
                            "token_ids",
                            "accepted_lengths",
                            "target_calls",
                            "draft_calls",
                        )
                    ):
                        raise ValueError(
                            "PARD observed replica differs from timed request"
                        )
                    raw, accepted = timed["token_ids"], timed["accepted_lengths"]
                    tokens = trim_pard_tokens(
                        raw, max_new_tokens=config["max_new_tokens"], eos_token_id=eos
                    )
                    seconds = timed["request_seconds"]
                    target_calls, draft_calls = (
                        timed["target_calls"],
                        timed["draft_calls"],
                    )
                if not tokens or (
                    len(tokens) < config["max_new_tokens"] and tokens[-1] != eos
                ):
                    raise ValueError("PARD campaign output ended without EOS or cap")
                row = {
                    "method": name,
                    "benchmark": "math500",
                    "problem_id": record["problem_id"],
                    "reference_answer": record.get("answer"),
                    "rank": rank,
                    "repetition": 0,
                    "turn_index": 0,
                    "input_ids": ids[0].tolist(),
                    "input_tokens": ids.shape[1],
                    "output_ids": tokens,
                    "raw_output_ids": raw,
                    "output_tokens": len(tokens),
                    "output_hash": hashlib.sha256(
                        torch.tensor(tokens, dtype=torch.int32).numpy().tobytes()
                    ).hexdigest(),
                    "completion": infer.tokenizer.decode(
                        tokens, skip_special_tokens=True
                    ),
                    "request_seconds": seconds,
                    "acceptance_lengths": accepted,
                    "accepted_draft_lengths": [n - 1 for n in accepted],
                    "proposal_lengths": [config["draft_k"]] * len(accepted),
                    "acceptance_length": sum(accepted) / len(accepted)
                    if accepted
                    else 1,
                    "target_calls": target_calls,
                    "draft_calls": draft_calls,
                    "verification_replica": proof,
                }
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
    if any(p._version != v or p.data_ptr() != ptr for p, v, ptr in frozen):
        raise ValueError("PARD campaign changed model parameters")
    (output / f"campaign-rank{rank}.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "rank": rank,
                "config_sha256": digest(args.config),
                "frozen_parameter_versions_and_storage_unchanged": True,
                "upstream_source_sha256": digest(source),
                "scope": config["scope"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
