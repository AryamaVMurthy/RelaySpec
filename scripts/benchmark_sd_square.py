"""Compare every frozen SD-square rate on common pretokenized requests."""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
from pilot_sd_square import cached_ar, digest

from relayspec.benchmarking import benchmark_turns
from relayspec.sd_square_adapter import (
    committed_tokens,
    finalize_sd_square_trace,
    load_inference_steering,
    load_sd_square,
    termination_status,
)


@torch.no_grad()
def generate(model, ids, cap, deferred, require_complete=True):
    model._relayspec_trace = []
    model._relayspec_tensor_trace = []
    model._relayspec_capture_tensors = deferred
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=cap)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    trace = finalize_sd_square_trace(model) if deferred else model._relayspec_trace
    tokens, raw = committed_tokens(trace, cap, model.eot_id)
    if not tokens or (
        require_complete and len(tokens) < cap and tokens[-1] != model.eot_id
    ):
        raise ValueError("SD-square output stopped before EOS or its declared cap")
    return tokens, raw, trace, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source_path = Path(config["source_config"])
    source = json.loads(source_path.read_text())
    setup_path = Path(os.environ["SD_SQUARE_SETUP_GATE"])
    setup = json.loads(setup_path.read_text())
    protocol_path = Path(config["verification_protocol"])
    protocol = json.loads(protocol_path.read_text())
    for path, sha in config.get("prerequisites", {}).items():
        prerequisite = Path(path)
        if (
            digest(prerequisite) != sha
            or json.loads(prerequisite.read_text())["status"] != "complete"
        ):
            raise ValueError("SD-square longer-output prerequisite did not pass")
    if (
        digest(source_path) != config["source_config_sha256"]
        or setup["status"] != "pass"
        or setup["config_sha256"] != digest(source_path)
        or digest(protocol_path) != config["verification_protocol_sha256"]
        or any(digest(Path(p)) != sha for p, sha in protocol["input_sha256"].items())
    ):
        raise ValueError("SD-square campaign source or protocol changed")
    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4 or torch.cuda.device_count() != 4:
        raise ValueError("SD-square campaign requires four GPU workers")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("SD-square campaign workspace changed")
    torch.cuda.set_device(rank)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(config["seed"])
    device = torch.device(f"cuda:{rank}")
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    upstream = load_sd_square("vendor/sd-square", source, setup["models"])
    settings = {
        k: v
        for k, v in source["training"].items()
        if k not in {"pool_examples", "max_length"}
    }
    model = upstream.TrainingModule(
        verifier=source["target"]["id"].lower(),
        drafter=source["drafter"]["id"].lower(),
        method="guided-drafter",
        greedy_sample=True,
        **settings,
    ).to(device)
    # Public eval.py casts these modules after loading the FP32 training state.
    # The inherited verifier stays FP16, including for runtime-local AR.
    model.d_base.to(torch.bfloat16)
    model.latent_mod_prep.to(torch.bfloat16)
    model.guidance_embd_layer.to(torch.bfloat16)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    initial = {n: p.detach().cpu().clone() for n, p in named}
    named_ids = {id(p) for _, p in named}
    frozen = [
        (p, p._version, p.data_ptr())
        for p in model.parameters()
        if id(p) not in named_ids
    ]
    guidance = model.v_base.get_decoder().guidance_embd_layer
    if (
        config["inference_precision"]
        != {
            "target": "float16",
            "drafter": "bfloat16",
            "steering": "bfloat16",
            "autocast": "bfloat16",
        }
        or any(p.dtype != torch.bfloat16 for p in model.d_base.parameters())
        or any(
            p.dtype != torch.float16
            for p in model.v_base.parameters()
            if id(p) not in named_ids
        )
    ):
        raise ValueError("SD-square public inference precision differs")
    model.eval().requires_grad_(False)
    records = [
        r
        for r in json.loads(Path(config["manifest_path"]).read_text())["records"]
        if r["benchmark"] == "math500"
    ]
    random.Random(config["seed"]).shuffle(records)
    start = config["request_offset"]
    records = records[start : start + config["requests"]][rank::4]
    prompts = []
    for record in records:
        ids = model.tok.apply_chat_template(
            [{"role": "user", "content": benchmark_turns(record)[0]}],
            enable_thinking=False,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        prompts.append((record, ids))
    if len(prompts) != config["requests"] // 4:
        raise ValueError("SD-square worker lacks declared prompts")
    validated, fingerprints = {}, {}

    def select(name):
        value = config["variants"][name]
        model.method = (
            "independent-drafter"
            if value["kind"] == "independent"
            else "guided-drafter"
        )
        model.v_base.get_decoder().guidance_embd_layer = (
            None if value["kind"] == "independent" else guidance
        )
        if value["kind"] in {"ar", "independent"}:
            return
        saved = (
            torch.load(
                value["checkpoint"], map_location="cpu", weights_only=True, mmap=True
            )["parameters"]
            if value["kind"] == "steering"
            else initial
        )
        identity = load_inference_steering(named, saved, value.get("trainable_sha256"))
        if (
            value.get("inference_sha256", identity["inference_sha256"])
            != identity["inference_sha256"]
        ):
            raise ValueError(
                "SD-square quality used different inference weights from selection"
            )
        if name in fingerprints and fingerprints[name] != identity:
            raise ValueError("SD-square steering identity changed between requests")
        fingerprints[name] = identity

    methods = list(config["variants"])
    if config.get("phase") == "termination_diagnostic":
        diagnostic = config["termination_diagnostic"]
        if (
            digest(Path(diagnostic["failure_registry"]))
            != diagnostic["failure_registry_sha256"]
        ):
            raise ValueError("termination diagnosis prerequisite changed")
        probes = []
        for name in methods:
            if config["variants"][name]["kind"] == "ar":
                continue
            select(name)
            for record, ids in prompts:
                for cap in (16, 64):
                    immediate = generate(model, ids, cap, False, require_complete=False)
                    deferred = generate(model, ids, cap, True, require_complete=False)
                    if immediate[:3] != deferred[:3]:
                        raise ValueError("termination probe observer changes decisions")
                    probes.append(
                        {
                            "method": name,
                            "problem_id": record["problem_id"],
                            "input_ids": ids[0].tolist(),
                            "trace": immediate[2],
                            "deferred_trace": deferred[2],
                            "observer_exact": True,
                            **termination_status(
                                immediate[2], cap, model.eot_id, model.NG
                            ),
                        }
                    )
        if any(
            p._version != version or p.data_ptr() != pointer
            for p, version, pointer in frozen
        ):
            raise ValueError("termination probe changed inherited weights")
        (output / f"termination-rank{rank}.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "rank": rank,
                    "config_sha256": digest(args.config),
                    "steering_fingerprints": fingerprints,
                    "probes": probes,
                    "frozen_parameter_versions_and_storage_unchanged": True,
                    "scope": "Unmodified public termination diagnosis, not measured campaign quality or completion.",
                },
                indent=2,
            )
            + "\n"
        )
        return
    for name in methods:
        if config["variants"][name]["kind"] == "ar":
            continue
        select(name)
        immediate = generate(
            model, prompts[0][1], config["observer_pilot_tokens"], False
        )
        deferred = generate(model, prompts[0][1], config["observer_pilot_tokens"], True)
        if immediate[:3] != deferred[:3]:
            raise ValueError("deferred SD-square observer changes tokens or acceptance")
        validated[name] = {
            "status": "pass",
            "tokens": immediate[0],
            "trace": immediate[2],
            "deferred_tokens": deferred[0],
            "deferred_trace": deferred[2],
        }
    with (output / f"benchmark-rank{rank}.jsonl").open("w") as stream:
        for prompt_index, (record, ids) in enumerate(prompts):
            order = methods.copy()
            random.Random(config["seed"] + rank * 4 + prompt_index).shuffle(order)
            for name in order:
                select(name)
                value = config["variants"][name]
                if value["kind"] == "ar":
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        cached_ar(
                            model.v_base, ids, config["warmup_tokens"], model.eot_id
                        )
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        tokens = cached_ar(
                            model.v_base, ids, config["max_new_tokens"], model.eot_id
                        )
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - started
                    raw, trace = tokens, []
                else:
                    generate(model, ids, config["warmup_tokens"], True)
                    tokens, raw, trace, seconds = generate(
                        model, ids, config["max_new_tokens"], True
                    )
                accepted = [b["accepted_draft_tokens"] for b in trace]
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
                    "completion": model.tok.decode(tokens, skip_special_tokens=True),
                    "request_seconds": seconds,
                    "acceptance_lengths": [n + 1 for n in accepted],
                    "accepted_draft_lengths": accepted,
                    "proposal_lengths": [model.NG] * len(accepted),
                    "acceptance_length": sum(n + 1 for n in accepted) / len(accepted)
                    if accepted
                    else 1,
                    "target_calls": len(accepted) + 1 if accepted else len(tokens),
                    "draft_calls": len(accepted) * model.NG + 1 if accepted else 0,
                    "verifier_trace": trace,
                    "steering_checkpoint_sha256": value.get("checkpoint_sha256"),
                    "steering_trainable_sha256": value.get("trainable_sha256"),
                    "steering_inference_sha256": fingerprints.get(name, {}).get(
                        "inference_sha256"
                    ),
                }
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
    if any(
        p._version != version or p.data_ptr() != pointer
        for p, version, pointer in frozen
    ):
        raise ValueError("SD-square campaign changed inherited weights")
    (output / f"campaign-rank{rank}.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "rank": rank,
                "config_sha256": digest(args.config),
                "observer_equality": validated,
                "steering_fingerprints": fingerprints,
                "inference_precision": config["inference_precision"],
                "frozen_parameter_versions_and_storage_unchanged": True,
                "scope": config["scope"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
