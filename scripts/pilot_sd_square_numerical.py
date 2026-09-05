"""Replay SD-square's failed AR comparison and observe both sides of the first difference."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import torch
from pilot_sd_square import cached_ar, digest, tensor_digest

from relayspec.sd_square_adapter import committed_tokens, load_sd_square


def install_causal_probe(model, spec):
    probe = {}

    def equal_caches(first, second):
        pairs = list(zip(first, second, strict=True))
        return all(
            torch.equal(a, b)
            for left, right in pairs
            for a, b in zip(left, right, strict=True)
        )

    def hook(module, args, kwargs):
        cache = kwargs.get("past_key_values")
        if (
            probe
            or cache is None
            or cache.get_seq_length() != spec["physical_cache_prefix"]
        ):
            return
        query = args[0] if args else kwargs["input_ids"]
        offset = spec["query_offset"]
        if query.shape[1] != 9 or not 0 <= offset < query.shape[1] - 1:
            raise ValueError("causal probe query differs from failed verification call")
        pristine = copy.deepcopy(cache)
        baseline_cache, changed_cache = copy.deepcopy(cache), copy.deepcopy(cache)
        if not equal_caches(cache, baseline_cache) or not equal_caches(
            cache, changed_cache
        ):
            raise ValueError("causal probe cache copies differ before intervention")
        if any(
            a.data_ptr() == b.data_ptr()
            for left, right in zip(cache, baseline_cache, strict=True)
            for a, b in zip(left, right, strict=True)
        ):
            raise ValueError("causal probe cache copy shares original storage")
        baseline_kwargs = dict(kwargs, past_key_values=baseline_cache)
        baseline = module.forward(*args, **baseline_kwargs)
        baseline_logits = model.v_base.lm_head(baseline["out"].last_hidden_state)
        changed = query.clone()
        changed[:, offset + 1 :] = (
            changed[:, offset + 1 :] + 7919
        ) % model.v_base.config.vocab_size
        changed_kwargs = dict(kwargs, past_key_values=changed_cache)
        changed_args = (changed, *args[1:]) if args else args
        if not args:
            changed_kwargs["input_ids"] = changed
        intervention = module.forward(*changed_args, **changed_kwargs)
        changed_logits = model.v_base.lm_head(intervention["out"].last_hidden_state)
        if not equal_caches(cache, pristine):
            raise ValueError("causal intervention changed the original cache")
        if not torch.equal(
            baseline_logits[:, : offset + 1], changed_logits[:, : offset + 1]
        ):
            raise ValueError("future query tokens changed earlier verifier logits")
        probe.update(
            status="pass",
            physical_cache_prefix=cache.get_seq_length(),
            query_offset=offset,
            original_query_ids=query[0].tolist(),
            changed_query_ids=changed[0].tolist(),
            cache_copies_exact_and_independent=True,
            original_cache_unchanged=True,
            prefix_logits_bit_identical=True,
            baseline_logits_float32_sha256=hashlib.sha256(
                baseline_logits[0].float().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
        )

    handle = model.v_base.get_decoder().register_forward_pre_hook(
        hook, with_kwargs=True
    )
    return probe, handle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    if "causal_probe" in spec:
        prior_numerical = (
            Path(os.environ["SD_SQUARE_NUMERICAL_OUTPUT"])
            / "sd-square-numerical-gate.json"
        )
        if (
            digest(prior_numerical)
            != spec["causal_probe"]["source_numerical_gate_sha256"]
        ):
            raise ValueError("causal probe numerical prerequisite changed")
    pilot_path = Path(spec["pilot_config"])
    pilot = json.loads(pilot_path.read_text())
    config_path = Path(pilot["setup_config"])
    config = json.loads(config_path.read_text())
    prior = Path(os.environ["SD_SQUARE_PRIOR_OUTPUT"])
    prior_gate = prior / "pilot-rank2.json"
    prior_row = prior / "decoding-rank2.json"
    setup_path = Path(os.environ["SD_SQUARE_SETUP_GATE"])
    setup = json.loads(setup_path.read_text())
    if (
        digest(pilot_path) != spec["pilot_config_sha256"]
        or digest(prior_gate) != spec["source_gate_sha256"]
        or digest(prior_row) != spec["source_decoding_sha256"]
        or setup["status"] != "pass"
        or setup["config_sha256"] != digest(config_path)
        or digest(config_path) != pilot["setup_config_sha256"]
    ):
        raise ValueError("SD-square replay prerequisite changed")
    gate, row = json.loads(prior_gate.read_text()), json.loads(prior_row.read_text())
    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4 or torch.cuda.device_count() != 4:
        raise ValueError("SD-square numerical replay requires four GPUs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("deterministic workspace changed")
    torch.cuda.set_device(rank)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(pilot["seed"])
    device = torch.device(f"cuda:{rank}")
    upstream = load_sd_square("vendor/sd-square", config, setup["models"])
    settings = {
        k: v
        for k, v in config["training"].items()
        if k not in {"pool_examples", "max_length"}
    }
    model = upstream.TrainingModule(
        verifier=config["target"]["id"].lower(),
        drafter=config["drafter"]["id"].lower(),
        method="guided-drafter",
        greedy_sample=True,
        loss_method=gate["objective"],
        **settings,
    ).to(device)
    checkpoint = Path(spec["checkpoint"])
    if digest(checkpoint) != spec["checkpoint_sha256"]:
        raise ValueError("SD-square replay steering checkpoint changed")
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if {n for n, _ in named} != set(saved["parameters"]):
        raise ValueError("SD-square replay steering architecture changed")
    with torch.no_grad():
        for name, parameter in named:
            parameter.copy_(saved["parameters"][name])
    if tensor_digest(named) != gate["trainable_sha256"]:
        raise ValueError("SD-square replay steering weights changed")
    del saved
    model.eval().requires_grad_(False)
    attention = spec["target_attention_by_rank"][rank]
    model.v_base.config._attn_implementation = attention
    if model.v_base.get_decoder().config._attn_implementation != attention:
        raise ValueError("target attention intervention did not propagate")
    ids = torch.tensor([row["input_ids"]], device=device)
    ar_trace = []
    model._relayspec_trace = []
    model._relayspec_detailed_trace = True
    probe, handle = (
        install_causal_probe(model, spec["causal_probe"])
        if "causal_probe" in spec
        else ({}, None)
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ar = cached_ar(model.v_base, ids, 64, model.eot_id, ar_trace)
        model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=64)
        public_ar, _ = model._generate_vanilla(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=16
        )
    if handle is not None:
        handle.remove()
        if probe.get("status") != "pass":
            raise ValueError("declared causal intervention did not execute")
        actual = next(
            b
            for b in model._relayspec_trace
            if b["physical_cache_prefix"] == probe["physical_cache_prefix"]
        )
        if actual["logits_float32_sha256"] != probe["baseline_logits_float32_sha256"]:
            raise ValueError(
                "copied-cache baseline differs from actual verifier logits"
            )
    tokens, raw = committed_tokens(model._relayspec_trace, 64, model.eot_id)
    if attention == "sdpa" and (tokens != row["output_ids"] or ar != row["ar_ids"]):
        raise ValueError("original SD-square numerical difference did not reproduce")
    difference = next((i for i, (a, b) in enumerate(zip(ar, tokens)) if a != b), None)
    detail = None
    if difference is not None:
        block = next(
            b
            for b in model._relayspec_trace
            if b["output_start"] <= difference < b["output_start"] + len(b["tokens"])
        )
        offset = difference - block["output_start"]
        query_prefix = (
            block["valid_cached_prefix_ids"] + block["query_ids"][: offset + 1]
        )
        expected_prefix = row["input_ids"] + tokens[:difference]
        if query_prefix != expected_prefix:
            raise ValueError(
                "SD-square target did not verify the same effective token prefix as AR"
            )
        if block["query_positions"][offset] != len(expected_prefix) - 1:
            raise ValueError("SD-square verifier uses inconsistent logical position")
        detail = {
            "index": difference,
            "ar_token": ar[difference],
            "sd_token": tokens[difference],
            "ar_top_ids": ar_trace[difference]["top_ids"],
            "ar_top_scores": ar_trace[difference]["top_scores"],
            "sd_top_ids": block["top_ids"][offset],
            "sd_top_scores": block["top_scores"][offset],
            "same_effective_token_prefix": True,
            "prefix_length": len(expected_prefix),
            "physical_cache_prefix": block["physical_cache_prefix"],
        }
    result = {
        "status": "pass",
        "rank": rank,
        "target_attention": attention,
        "config_sha256": digest(args.config),
        "source_gate_sha256": digest(prior_gate),
        "source_decoding_sha256": digest(prior_row),
        "ar_ids": ar,
        "sd_ids": tokens,
        "ar_trace": ar_trace,
        "sd_trace": model._relayspec_trace,
        "raw_committed_ids": raw,
        "first_difference": detail,
        "public_ar_first16": public_ar[0, ids.shape[1] :].tolist(),
        "public_ar_matches_correct_cached_ar": public_ar[0, ids.shape[1] :].tolist()
        == ar[:16],
        "scope": spec["scope"],
    }
    if handle is not None:
        result["causal_probe"] = probe
    (Path(os.environ["RELAYSPEC_OUTPUT"]) / f"numerical-rank{rank}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
