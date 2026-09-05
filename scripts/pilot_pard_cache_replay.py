"""Separate query-shape and cached-prefix effects at PARD's first difference."""

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import transformers
import yaml
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, StaticCache


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(logits):
    logits = logits.float()
    values, indices = logits.topk(5, dim=-1)
    return {
        "argmax_ids": logits.argmax(-1).cpu().tolist(),
        "top_ids": indices.cpu().tolist(),
        "top_scores": values.cpu().tolist(),
    }


def cache_difference(left, right, length=None):
    maximum = 0.0
    tensors = 0
    for attribute in ("key_cache", "value_cache"):
        a, b = getattr(left, attribute), getattr(right, attribute)
        if len(a) != len(b) or not a:
            raise ValueError("cache layer count differs")
        for x, y in zip(a, b, strict=True):
            if length is not None:
                x, y = x[:, :, :length], y[:, :, :length]
            if x.shape != y.shape:
                raise ValueError("cache shape differs")
            maximum = max(maximum, float((x.float() - y.float()).abs().max()))
            tensors += 1
    return {
        "exact": maximum == 0.0,
        "max_absolute_difference": maximum,
        "tensors_compared": tensors,
        "prefix_length": length,
    }


def checked_copy(cache):
    result = copy.deepcopy(cache)
    if not cache_difference(cache, result)["exact"]:
        raise ValueError("cache clone is not identical")
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--declaration", type=Path, required=True)
    args = parser.parse_args()
    declaration = json.loads(args.declaration.read_text())
    config_path = Path("configs/submission/baselines/pard-pilot.yaml")
    config = yaml.safe_load(config_path.read_text())
    setup_path = Path(os.environ["PARD_SETUP_GATE"])
    setup = json.loads(setup_path.read_text())
    if setup["status"] != "pass" or setup["config_sha256"] != digest(config_path):
        raise ValueError("original model setup is not verified")
    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4 or torch.cuda.device_count() != 4:
        raise ValueError("diagnostic requires four independent GPU workers")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    query_length = declaration["query_lengths"][rank]
    offset = declaration["offset"]
    trace = declaration["pard_target_trace"][-1]
    base_position = trace["cache_position"][0]
    prefix = trace["prefix_ids"]
    if (
        prefix
        != declaration["input_ids"]
        + declaration["ar_token_ids"][: trace["output_start"]]
    ):
        raise ValueError("diagnostic prefix is not shared with AR")
    if trace["incoming_ids"][: offset + 1] != (
        prefix[-1:]
        + declaration["ar_token_ids"][
            trace["output_start"] : declaration["first_difference"]
        ]
    ):
        raise ValueError("verification input preceding the difference is inconsistent")
    path = snapshot_download(
        repo_id=config["target"]["id"],
        revision=config["target"]["revision"],
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
        local_files_only=True,
    )
    started = time.perf_counter()
    model = (
        AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )

    def new_cache():
        return StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=declaration["max_cache_len"],
            device=device,
            dtype=model.dtype,
        )

    def forward(cache, ids, positions):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            return model(
                input_ids=torch.tensor([ids], device=device),
                position_ids=None,
                past_key_values=cache,
                cache_position=torch.tensor(positions, device=device),
                use_cache=True,
                attention_mask=None,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            ).logits[0]

    ar_cache = new_cache()
    prompt = declaration["input_ids"]
    ar_matches = []
    logits = forward(ar_cache, prompt, list(range(len(prompt))))
    ar_matches.append(int(logits[-1].argmax()) == declaration["ar_token_ids"][0])
    for i, token in enumerate(prefix[len(prompt) : -1]):
        logits = forward(ar_cache, [token], [len(prompt) + i])
        ar_matches.append(
            int(logits[-1].argmax()) == declaration["ar_token_ids"][i + 1]
        )
    pard_cache = new_cache()
    pard_matches = []
    for previous in declaration["pard_target_trace"][:-1]:
        logits = forward(
            pard_cache, previous["incoming_ids"], previous["cache_position"]
        )
        observed = scores(logits[-len(previous["argmax_ids"]) :])
        pard_matches.append(all(observed[k] == previous[k] for k in observed))
    prefix_comparison = cache_difference(ar_cache, pard_cache, base_position)
    incoming = list(trace["incoming_ids"][:query_length])
    incoming += [0] * (query_length - len(incoming))
    positions = list(range(base_position, base_position + query_length))
    results = {}
    chosen_logits = {}
    for name, base in [
        ("ar_prefix_cache", ar_cache),
        ("pard_prefix_cache", pard_cache),
    ]:
        serial_cache = checked_copy(base)
        serial = []
        for token, position in zip(
            incoming[: offset + 1], positions[: offset + 1], strict=True
        ):
            serial.append(forward(serial_cache, [token], [position])[-1])
        serial = torch.stack(serial)
        del serial_cache
        block_cache = checked_copy(base)
        block = forward(block_cache, incoming, positions)[: offset + 1]
        del block_cache
        alternate = incoming[: offset + 1] + [
            (t + 123) % model.config.vocab_size for t in incoming[offset + 1 :]
        ]
        alternate_cache = checked_copy(base)
        changed = forward(alternate_cache, alternate, positions)[: offset + 1]
        del alternate_cache
        chosen_logits[name] = {"serial": serial[-1], "block": block[-1]}
        results[name] = {
            "cache_copies_verified_exact": True,
            "serial": scores(serial),
            "block": scores(block),
            "serial_block_max_logit_difference": float(
                (serial.float() - block.float()).abs().max()
            ),
            "causal_future_tokens_changed": query_length - offset - 1,
            "causal_prefix_exact": torch.equal(block, changed),
            "causal_prefix_max_logit_difference": float(
                (block.float() - changed.float()).abs().max()
            ),
        }
    cross = {
        mode: float(
            (
                chosen_logits["ar_prefix_cache"][mode].float()
                - chosen_logits["pard_prefix_cache"][mode].float()
            )
            .abs()
            .max()
        )
        for mode in ("serial", "block")
    }
    original_pard = {k: trace[k][offset] for k in ("top_ids", "top_scores")}
    original_pard["argmax_id"] = trace["argmax_ids"][offset]
    original_ar = declaration["ar_target_trace"][declaration["first_difference"]]
    ar_final = results["ar_prefix_cache"]["serial"]
    ar_final_exact = (
        ar_final["argmax_ids"][-1] == original_ar["argmax_id"]
        and ar_final["top_ids"][-1] == original_ar["top_ids"]
        and ar_final["top_scores"][-1] == original_ar["top_scores"]
    )
    pard_final = results["pard_prefix_cache"]["block"]
    pard_final_exact = (
        pard_final["argmax_ids"][-1] == original_pard["argmax_id"]
        and pard_final["top_ids"][-1] == original_pard["top_ids"]
        and pard_final["top_scores"][-1] == original_pard["top_scores"]
    )
    valid = (
        all(ar_matches)
        and all(pard_matches)
        and ar_final_exact
        and (query_length != len(trace["incoming_ids"]) or pard_final_exact)
        and all(v["causal_prefix_exact"] for v in results.values())
    )
    torch.cuda.synchronize()
    result = {
        "status": "pass" if valid else "fail",
        "rank": rank,
        "query_length": query_length,
        "offset": offset,
        "input_sha256": {
            str(args.declaration): digest(args.declaration),
            str(config_path): digest(config_path),
            "setup_gate": digest(setup_path),
        },
        "source_rows_sha256": declaration["source_rows_sha256"],
        "target": config["target"],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(),
        "elapsed_seconds": time.perf_counter() - started,
        "ar_replayed_calls": len(ar_matches),
        "ar_replay_argmax_exact": all(ar_matches),
        "pard_replayed_calls": len(pard_matches),
        "pard_replay_top5_exact": all(pard_matches),
        "ar_final_top5_exact": ar_final_exact,
        "pard_original_shape_final_top5_exact": pard_final_exact
        if query_length == len(trace["incoming_ids"])
        else None,
        "committed_cache_comparison": prefix_comparison,
        "comparisons": results,
        "cache_effect_at_difference_max_logit_difference": cross,
        "original_ar": original_ar,
        "original_pard": original_pard,
        "scope": declaration["scope"],
    }
    (Path(os.environ["RELAYSPEC_OUTPUT"]) / f"cache-replay-rank{rank}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"rank": rank, "status": result["status"], "query_length": query_length}
        )
    )


if __name__ == "__main__":
    main()
