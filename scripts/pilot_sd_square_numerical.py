"""Replay SD-square's failed AR comparison and observe both sides of the first difference."""

import argparse
import json
import os
from pathlib import Path

import torch
from pilot_sd_square import cached_ar, digest, tensor_digest

from relayspec.sd_square_adapter import committed_tokens, load_sd_square


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
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
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ar = cached_ar(model.v_base, ids, 64, model.eot_id, ar_trace)
        model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=64)
        public_ar, _ = model._generate_vanilla(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=16
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
    (Path(os.environ["RELAYSPEC_OUTPUT"]) / f"numerical-rank{rank}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
