"""Exercise public frozen SD-square fitting, reload and greedy decoding on four GPUs."""

import argparse
import hashlib
import json
import os
import random
import time
import traceback
from pathlib import Path

import torch

from relayspec.benchmarking import benchmark_turns
from relayspec.sd_square_adapter import committed_tokens, load_sd_square


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def tensor_digest(named):
    result = hashlib.sha256()
    for name, value in named:
        result.update(name.encode())
        result.update(str((tuple(value.shape), value.dtype)).encode())
        result.update(
            value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return result.hexdigest()


@torch.no_grad()
def cached_ar(model, ids, cap, eos, trace=None):
    from transformers import DynamicCache

    cache, current, result = DynamicCache(), ids, []
    for _ in range(cap):
        out = model(
            input_ids=current, past_key_values=cache, use_cache=True, return_dict=True
        )
        current = out.logits[:, -1:].argmax(-1)
        if trace is not None:
            values, indices = out.logits[0, -1].float().topk(5)
            trace.append(
                {
                    "output_start": len(result),
                    "argmax_id": int(current.item()),
                    "top_ids": indices.cpu().tolist(),
                    "top_scores": values.cpu().tolist(),
                }
            )
        result.append(int(current.item()))
        if result[-1] == eos:
            break
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    pilot = json.loads(args.config.read_text())
    config_path = Path(pilot["setup_config"])
    config = json.loads(config_path.read_text())
    setup_path = Path(os.environ["SD_SQUARE_SETUP_GATE"])
    setup = json.loads(setup_path.read_text())
    if (
        digest(config_path) != pilot["setup_config_sha256"]
        or setup["status"] != "pass"
        or setup["config_sha256"] != digest(config_path)
    ):
        raise ValueError("SD-square pilot differs from verified setup")
    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4 or torch.cuda.device_count() != 4:
        raise ValueError("SD-square pilot requires four GPU workers")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("declared deterministic workspace is required")
    torch.cuda.set_device(rank)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(pilot["seed"])
    torch.cuda.manual_seed_all(pilot["seed"])
    device = torch.device(f"cuda:{rank}")
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    fitting = Path(os.environ["RELAYSPEC_FIT_OUTPUT"]) / f"rank{rank}"
    fitting.mkdir(parents=True, exist_ok=False)
    gate = {
        "status": "fail",
        "phase": "load",
        "rank": rank,
        "config_sha256": digest(args.config),
        "setup_gate_sha256": digest(setup_path),
        "objective": pilot["worker_objectives"][rank],
        "seed": pilot["seed"],
    }
    started = time.perf_counter()
    try:
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
        gate["model_load_seconds"] = time.perf_counter() - started
        model.v_base.eval()
        model.d_base.train()
        params = list(model.latent_mod_prep.parameters()) + list(
            model.guidance_embd_layer.parameters()
        )
        ids_trainable = {id(p) for p in params}
        if len(ids_trainable) != len(params) or any(
            p.requires_grad != (id(p) in ids_trainable) for p in model.parameters()
        ):
            raise ValueError(
                "SD-square frozen training has undeclared trainable parameters"
            )
        frozen = [
            (name, p, p._version, p.data_ptr())
            for name, p in model.named_parameters()
            if id(p) not in ids_trainable
        ]
        draft_before = tensor_digest(model.d_base.named_parameters())
        gate["trainable_parameters"] = sum(p.numel() for p in params)
        gate["trainable_names"] = [
            n for n, p in model.named_parameters() if id(p) in ids_trainable
        ]
        gate["attention"] = {
            "target": model.v_base.config._attn_implementation,
            "drafter": model.d_base.config._attn_implementation,
        }
        gate["storage_dtypes"] = {
            "target": str(next(model.v_base.parameters()).dtype),
            "drafter": str(next(model.d_base.parameters()).dtype),
        }
        cache = Path(os.environ["RELAYSPEC_FEATURE_CACHE"])
        index_path = cache / "cache-index.json"
        index = json.loads(index_path.read_text())
        if (
            digest(index_path) != pilot["feature_cache_index_sha256"]
            or index["status"] != "pass"
            or index["metadata"]["config"]["target"] != config["target"]
        ):
            raise ValueError("SD-square pilot data cache differs from declaration")
        entries = [r for r in index["entries"] if r["split"] == "train"][:512]
        if len(entries) != 512 or len({r["file"] for r in entries}) != 512:
            raise ValueError("SD-square pilot requires declared512-example pool")
        gate["ordered_pool_files"] = [r["file"] for r in entries]
        records = []
        for entry in entries[: pilot["updates"]]:
            path = cache / entry["file"]
            if digest(path) != entry["sha256"]:
                raise ValueError("SD-square pilot input record changed")
            saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            records.append(saved["input_ids"].clone().to(device))
        gate["phase"] = "zero_guidance"
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            test = records[0]
            zero = model.latent_mod_prep(
                torch.zeros(1, test.shape[1], model.V_H_DIM, device=device)
            )
            a = model.d_base.get_decoder()(
                test, use_cache=False, return_dict=True
            ).last_hidden_state
            b = model.d_base.get_decoder()(
                test, guidance_embeds=zero, use_cache=False, return_dict=True
            ).last_hidden_state
            if not torch.equal(a, b):
                raise ValueError("zero-guidance SD-square drafter identity failed")
        del a, b, zero
        gate["zero_guidance_bit_identical"] = True
        from src.util import setup_optim

        optim = setup_optim(
            params,
            lr_start=pilot["learning_rate"],
            lr_end=pilot["learning_rate_end"],
            warmup_steps=pilot["warmup_steps"],
            estimated_stepping_batches=pilot["updates"],
        )
        optimizer, scheduler = optim["optimizer"], optim["lr_scheduler"]["scheduler"]
        gate["phase"] = "training"
        history = []
        for step, tokens in enumerate(records, 1):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model.process_batch(
                    {"targets": tokens}, log_extras=False, compute_tvd=True
                )
            loss.backward()
            if not torch.isfinite(loss) or any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in params
            ):
                raise ValueError("nonfinite SD-square loss or gradient")
            norm = torch.nn.utils.clip_grad_norm_(params, pilot["gradient_clip"])
            if float(norm) <= 0:
                raise ValueError("SD-square steering received no gradient")
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(norm),
                    "seconds": time.perf_counter() - tick,
                    "record": entries[step - 1]["file"],
                }
            )
            (output / f"training-rank{rank}.json").write_text(
                json.dumps(history, indent=2) + "\n"
            )
        if any(
            p.requires_grad
            or p.grad is not None
            or p._version != version
            or p.data_ptr() != pointer
            for _, p, version, pointer in frozen
        ):
            raise ValueError("SD-square changed a frozen parameter")
        draft_after = tensor_digest(model.d_base.named_parameters())
        if draft_after != draft_before:
            raise ValueError("frozen drafter weights changed")
        gate.update(
            frozen_parameter_versions_and_storage_unchanged=True,
            frozen_drafter_sha256=draft_after,
            training=history,
            distinct_records_seen=len(records),
            training_seconds=sum(r["seconds"] for r in history),
        )
        gate["phase"] = "checkpoint"
        named = [(n, p) for n, p in model.named_parameters() if id(p) in ids_trainable]
        checkpoint = fitting / "steering-state.pt"
        before = tensor_digest(named)
        torch.save(
            {
                "parameters": {n: p.detach().cpu() for n, p in named},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state(device),
                "config_sha256": digest(args.config),
                "source_commit": config["source_commit"],
            },
            checkpoint,
        )
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        with torch.no_grad():
            for name, parameter in named:
                parameter.zero_()
                parameter.copy_(saved["parameters"][name])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if tensor_digest(named) != before:
            raise ValueError("SD-square checkpoint reload changed steering")
        restored = optimizer.state_dict()
        if restored["param_groups"] != saved["optimizer"]["param_groups"]:
            raise ValueError("SD-square optimizer groups changed on reload")
        for key, values in saved["optimizer"]["state"].items():
            for field, value in values.items():
                actual = restored["state"][key][field]
                if isinstance(value, torch.Tensor):
                    if actual.dtype != value.dtype or not torch.equal(
                        actual.cpu(), value
                    ):
                        raise ValueError("SD-square optimizer tensor changed on reload")
                elif actual != value:
                    raise ValueError("SD-square optimizer value changed on reload")
        gate.update(
            checkpoint=str(checkpoint),
            checkpoint_sha256=digest(checkpoint),
            trainable_sha256=before,
            steering_reload_bit_identical=True,
            optimizer_reload_bit_identical=True,
        )
        del saved, optimizer, scheduler, optim, restored, loss, norm
        model.zero_grad(set_to_none=True)
        model.eval()
        torch.cuda.empty_cache()
        gate["phase"] = "decoding"
        manifest = json.loads(Path(pilot["manifest_path"]).read_text())
        problems = [r for r in manifest["records"] if r["benchmark"] == "math500"]
        random.Random(pilot["seed"]).shuffle(problems)
        problem = problems[rank // 2]
        ids = model.tok.apply_chat_template(
            [{"role": "user", "content": benchmark_turns(problem)[0]}],
            enable_thinking=False,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        model._relayspec_trace = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ar = cached_ar(model.v_base, ids, pilot["max_new_tokens"], model.eot_id)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=pilot["max_new_tokens"],
            )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - tick
        output_tokens, raw = committed_tokens(
            model._relayspec_trace, pilot["max_new_tokens"], model.eot_id
        )
        row = {
            "problem": problem,
            "input_ids": ids[0].tolist(),
            "output_ids": output_tokens,
            "raw_committed_ids": raw,
            "ar_ids": ar,
            "ar_exact": output_tokens == ar,
            "request_seconds_with_observer": seconds,
            "trace": model._relayspec_trace,
        }
        (output / f"decoding-rank{rank}.json").write_text(
            json.dumps(row, indent=2) + "\n"
        )
        gate.update(
            status="pass" if row["ar_exact"] else "fail",
            phase="complete",
            ar_exact=row["ar_exact"],
            peak_gpu_bytes=torch.cuda.max_memory_allocated(device),
            total_seconds=time.perf_counter() - started,
            scope=pilot["scope"],
        )
    except Exception:
        gate["error"] = traceback.format_exc()
        raise
    finally:
        (output / f"pilot-rank{rank}.json").write_text(
            json.dumps(gate, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
