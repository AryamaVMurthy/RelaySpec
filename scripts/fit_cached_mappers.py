"""Fit one independent mapper per GPU, using a common frozen-feature cache."""

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch

from relayspec.feature_cache import (
    MappedFeatureReader,
    load_cached_record,
    pad_feature_batch,
    record_weighted_loss,
    restore_output_norm,
)
from relayspec.fit_continuation import validate_continuation
from relayspec.fitting_validation import interface_diagnostics
from relayspec.losses import explicit_l2_penalty, interface_alignment_loss
from relayspec.relay import TargetFeatureRelay


def main():
    worker_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--equivalence-pilot", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4 or torch.cuda.device_count() != 4:
        raise ValueError("cached fitting campaign requires four GPU workers")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    trials = json.loads(args.trials.read_text())["trials"]
    if len(trials) != 4:
        raise ValueError("one four-GPU fitting batch must declare exactly four trials")
    trial = trials[rank]
    seed = int(trial["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    root = Path(os.environ["RELAYSPEC_FEATURE_CACHE"])
    index_path = root / "cache-index.json"
    index = json.loads(index_path.read_text())
    if index["status"] != "pass":
        raise ValueError("feature cache has not passed its completion gate")
    metadata = index["metadata"]
    entries = [e for e in index["entries"] if e["split"] == "train"]
    n = int(trial["distinct_examples"])
    steps = int(trial["steps"])
    if n < 4 or n % 4 or n > len(entries) or steps <= 0:
        raise ValueError("trial data/update budget is invalid")
    entries = entries[:n]
    l2 = float(trial.get("l2_weight", 0))
    decay = float(trial.get("weight_decay", 0))
    if any(not math.isfinite(x) or x < 0 for x in [l2, decay]) or (l2 and decay):
        raise ValueError("L2 and AdamW decay require separate finite nonnegative arms")
    width = trial.get("width")
    architecture = trial["architecture"]
    if architecture not in {"dense", "factorized", "mlp"}:
        raise ValueError("unknown cached mapper architecture")
    opts = (
        {"mlp_hidden_width": width}
        if architecture == "mlp"
        else {"factorized_rank": width}
        if architecture == "factorized"
        else {}
    )
    relay = TargetFeatureRelay(
        target_hidden_size=metadata["target_hidden_size"],
        num_taps=len(metadata["target_layer_ids"]),
        draft_hidden_size=metadata["draft_hidden_size"],
        eps=metadata["rms_norm_eps"],
        normalize_input=trial.get("normalize_input", metadata["family"] == "dflash"),
        **opts,
    ).to(device)
    norm = restore_output_norm(root, metadata, device)
    optimizer = torch.optim.AdamW(
        relay.parameters(), lr=float(trial["learning_rate"]), weight_decay=decay
    )
    output = Path(os.environ["RELAYSPEC_OUTPUT"]) / trial["name"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "trial.json").write_text(json.dumps(trial, indent=2) + "\n")
    objective = trial.get("feature_objective", "relative_interface_mse")
    cosine_weight = trial.get("historical_cosine_weight")
    setup_started = time.perf_counter()
    backend = trial.get("cache_backend", "buffer")
    if backend not in {"buffer", "mmap"}:
        raise ValueError("unknown feature-cache backend")
    reader = MappedFeatureReader(root) if backend == "mmap" else None

    def read_entry(entry):
        return reader(entry) if reader is not None else load_cached_record(root, entry)

    # Small fitting sets fit comfortably in host memory. Larger sets stream
    # through the shared OS page cache instead of duplicating hundreds of GB.
    preloaded = {e["index"]: read_entry(e) for e in entries} if n <= 2048 else {}
    device_cache = bool(trial.get("device_cache", False))
    if device_cache:
        free, _ = torch.cuda.mem_get_info(device)
        if sum(e["bytes"] for e in entries) > 0.7 * free:
            raise ValueError("declared GPU feature cache exceeds its memory budget")
        preloaded = {
            e["index"]: tuple(t.to(device) for t in read_entry(e)) for e in entries
        }

    def get_example(entry):
        return preloaded[entry["index"]] if preloaded else read_entry(entry)

    validation_count = int(trial.get("validation_records", 32))
    validation_entries = [e for e in index["entries"] if e["split"] == "validation"]
    if validation_count < 1 or validation_count > len(validation_entries):
        raise ValueError("cache cannot supply the declared validation count")
    validation_entries = validation_entries[:validation_count]
    validation = [read_entry(e) for e in validation_entries]
    diagnostic_train = [
        get_example(e)
        for e in entries[: int(trial.get("diagnostic_train_records", 32))]
    ]
    if not validation:
        raise ValueError("cached fitting requires a fixed validation split")
    setup_seconds = time.perf_counter() - setup_started
    if args.equivalence_pilot:
        examples = [get_example(e) for e in entries[:4]]
        reference = copy.deepcopy(relay)
        reference_losses = []
        for x, y in examples:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = norm(reference(x.to(device)))
            reference_losses.append(
                interface_alignment_loss(
                    predicted,
                    y.to(device),
                    objective=objective,
                    historical_cosine_weight=cosine_weight,
                )
            )
        reference_loss = torch.stack(reference_losses).mean()
        reference_loss.backward()
        x, y, mask = pad_feature_batch(examples, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = norm(relay(x))
        batched_loss = record_weighted_loss(
            prediction,
            y,
            mask,
            objective=objective,
            historical_cosine_weight=cosine_weight,
        )
        batched_loss.backward()
        ref_grad = torch.cat([p.grad.flatten() for p in reference.parameters()])
        batch_grad = torch.cat([p.grad.flatten() for p in relay.parameters()])
        relative_gradient_error = float(
            (ref_grad - batch_grad).norm() / ref_grad.norm().clamp_min(1e-12)
        )
        relative_loss_error = abs(float(reference_loss - batched_loss)) / max(
            abs(float(reference_loss)), 1e-12
        )
        if relative_gradient_error > 0.02 or relative_loss_error > 0.01:
            raise ValueError(
                f"BF16 batch equivalence exceeds declared tolerances: loss={relative_loss_error}, gradient={relative_gradient_error}"
            )
        (output / "equivalence-gate.json").write_text(
            json.dumps(
                {
                    "status": "pass",
                    "relative_gradient_error": relative_gradient_error,
                    "relative_loss_error": relative_loss_error,
                    "gradient_tolerance": 0.02,
                    "loss_tolerance": 0.01,
                    "scope": "BF16 single-record gradient mean versus padded batch four at identical weights; floating-point kernel shapes may differ.",
                },
                indent=2,
            )
            + "\n"
        )
        del (
            reference,
            reference_losses,
            reference_loss,
            batched_loss,
            ref_grad,
            batch_grad,
            prediction,
            predicted,
            x,
            y,
            mask,
        )
        optimizer.zero_grad(set_to_none=True)
    saves = sorted(
        set([int(v) for v in trial.get("checkpoint_steps", [steps])] + [steps])
    )
    if any(s < 1 or s > steps for s in saves):
        raise ValueError("checkpoint steps lie outside the fitting budget")

    def diagnostics(step):
        values = {
            name: interface_diagnostics(
                relay,
                examples,
                device=device,
                objective=objective,
                historical_cosine_weight=cosine_weight,
                output_transform=norm,
            )
            for name, examples in [
                ("train", diagnostic_train),
                ("validation", validation),
            ]
        }
        with (output / "validation.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {"step": step, "groups": values, "regularization_included": False}
                )
                + "\n"
            )

    start_step = 0
    tokens_seen = 0
    if trial.get("resume_from"):
        if args.equivalence_pilot:
            raise ValueError("batch-gradient pilot cannot also resume a fit")
        parent_path = Path(trial["resume_from"])
        if (
            hashlib.sha256(parent_path.read_bytes()).hexdigest()
            != trial["resume_sha256"]
        ):
            raise ValueError("continuation state hash mismatch")
        state = torch.load(parent_path, weights_only=True, map_location="cpu")
        start_step = validate_continuation(
            state, trial, hashlib.sha256(index_path.read_bytes()).hexdigest()
        )
        relay.load_state_dict(state["relay"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state(state["cuda_rng_state"], device)
        tokens_seen = state["tokens_seen"]
        del state
    initial_validation_started = time.perf_counter()
    diagnostics(start_step)
    initial_validation_seconds = time.perf_counter() - initial_validation_started
    started = time.perf_counter()
    validation_seconds = 0.0
    io_seconds = 0.0
    training_update_seconds = 0.0
    checkpoint_export_seconds = 0.0
    with (output / "training.jsonl").open("x") as log:
        for step in range(start_step + 1, steps + 1):
            update_started = time.perf_counter()
            io_start = time.perf_counter()
            selected = [entries[((step - 1) * 4 + j) % n] for j in range(4)]
            x, y, mask = pad_feature_batch([get_example(e) for e in selected], device)
            io_seconds += time.perf_counter() - io_start
            tokens_seen += sum(e["tokens"] for e in selected)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = norm(relay(x))
            feature_loss = record_weighted_loss(
                prediction,
                y,
                mask,
                objective=objective,
                historical_cosine_weight=cosine_weight,
            )
            penalty = explicit_l2_penalty(relay.parameters(), l2)
            loss = feature_loss + penalty
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                relay.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite cached fitting loss")
            log.write(
                json.dumps(
                    {
                        "step": step,
                        "feature_loss": float(feature_loss.detach()),
                        "l2_penalty": float(penalty.detach()),
                        "loss": float(loss.detach()),
                        "gradient_norm": float(gradient_norm),
                        "tokens_seen": tokens_seen,
                        "record_presentations": step * 4,
                        "distinct_records_seen": min(n, step * 4),
                        "epochs": step * 4 / n,
                    }
                )
                + "\n"
            )
            training_update_seconds += time.perf_counter() - update_started
            if step in saves:
                validation_started = time.perf_counter()
                diagnostics(step)
                validation_seconds += time.perf_counter() - validation_started
                export_started = time.perf_counter()
                checkpoint = {
                    "relay": relay.state_dict(),
                    "target_layer_ids": metadata["target_layer_ids"],
                    "steps": step,
                    "seed": seed,
                    "proposer_family": metadata["family"],
                    "relay_architecture": "normalized_linear"
                    if relay.normalize_input
                    else "scale_preserving_linear",
                    "feature_objective": objective,
                    **opts,
                    "data_accounting": {
                        "distinct_records": n,
                        "distinct_records_seen": min(n, step * 4),
                        "record_presentations": step * 4,
                        "epochs": step * 4 / n,
                    },
                    "feature_cache_index_sha256": hashlib.sha256(
                        index_path.read_bytes()
                    ).hexdigest(),
                    "initialization_protocol": "direct mapper seed; separate from legacy model-loading RNG consumption",
                }
                if any(not torch.isfinite(p).all() for p in relay.parameters()):
                    raise ValueError("nonfinite cached mapper checkpoint")
                torch.save(checkpoint, output / f"step-{step:06d}.pt")
                checkpoint_export_seconds += time.perf_counter() - export_started
            if step == 1 or step % 128 == 0:
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "trial": trial["name"],
                            "step": step,
                            "loss": float(loss.detach()),
                        }
                    ),
                    flush=True,
                )
    # Preserve the optimizer as well as the map so a still-improving endpoint
    # can be extended without restarting or silently resetting Adam moments.
    continuation = output / "continuation.pt"
    export_started = time.perf_counter()
    torch.save(
        {
            "relay": relay.state_dict(),
            "optimizer": optimizer.state_dict(),
            "completed_trial": trial,
            "steps": steps,
            "tokens_seen": tokens_seen,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device),
            "feature_cache_index_sha256": hashlib.sha256(
                index_path.read_bytes()
            ).hexdigest(),
        },
        continuation,
    )
    checkpoint_export_seconds += time.perf_counter() - export_started
    (output / "fit-complete.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "trial": trial,
                "parameters": sum(p.numel() for p in relay.parameters()),
                "steps": steps,
                "start_step": start_step,
                "updates_this_job": steps - start_step,
                "tokens_seen": tokens_seen,
                "record_presentations": steps * 4,
                "epochs": steps * 4 / n,
                "distinct_records_seen": min(n, steps * 4),
                "loop_seconds": time.perf_counter() - started,
                "training_update_seconds": training_update_seconds,
                "initial_validation_seconds": initial_validation_seconds,
                "checkpoint_export_seconds": checkpoint_export_seconds,
                "worker_total_seconds": time.perf_counter() - worker_started,
                "timing_scope": "Training-update time includes data access, loss/gradient "
                "computation, optimizer steps and training-log writes. It excludes "
                "validation and checkpoint export. Worker total includes model/cache "
                "setup and initial validation, but excludes Python import startup "
                "and this completion-record/hash write. Outer batch wall time records "
                "the full launched process cost.",
                "validation_seconds": validation_seconds,
                "input_io_seconds": io_seconds,
                "setup_seconds": setup_seconds,
                "cache_backend": backend,
                "device_cache": device_cache,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(device),
                "feature_cache_index_sha256": hashlib.sha256(
                    index_path.read_bytes()
                ).hexdigest(),
                "cache_extraction_metadata": metadata,
                "continuation_sha256": hashlib.sha256(
                    continuation.read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
