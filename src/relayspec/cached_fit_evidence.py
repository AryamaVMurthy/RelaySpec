"""Reconstruct fresh cached-fit accounting from its per-update and validation logs."""

import hashlib
import json
import math


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_fit_artifacts(folder, trial, *, cache_sha256):
    complete_path = folder / "fit-complete.json"
    validation_path = folder / "validation.jsonl"
    training_path = folder / "training.jsonl"
    trial_path = folder / "trial.json"
    complete = json.loads(complete_path.read_text())
    if complete.get("status") != "pass" or complete["trial"] != trial:
        raise ValueError("fit does not match declared trial")
    if json.loads(trial_path.read_text()) != trial:
        raise ValueError("saved trial differs from declared trial")
    if complete["feature_cache_index_sha256"] != cache_sha256:
        raise ValueError("fit used a different frozen-feature cache")
    if complete["start_step"] != 0 or trial.get("resume_from"):
        raise ValueError("this audit requires a fresh fitting trajectory")
    n, steps = trial["distinct_examples"], trial["steps"]
    expected_counts = {
        "steps": steps,
        "updates_this_job": steps,
        "record_presentations": steps * 4,
        "epochs": steps * 4 / n,
        "distinct_records_seen": min(n, steps * 4),
    }
    if any(complete[key] != value for key, value in expected_counts.items()):
        raise ValueError("fit completion miscounts training exposure")
    training = [json.loads(line) for line in training_path.read_text().splitlines()]
    if [row["step"] for row in training] != list(range(1, steps + 1)):
        raise ValueError("training log has missing or duplicate updates")
    prior_tokens = 0
    for row in training:
        step = row["step"]
        if (
            row["record_presentations"] != step * 4
            or row["distinct_records_seen"] != min(n, step * 4)
            or row["epochs"] != step * 4 / n
            or not isinstance(row["tokens_seen"], int)
            or row["tokens_seen"] <= prior_tokens
        ):
            raise ValueError("training log miscounts record/token exposure")
        for key in ["feature_loss", "l2_penalty", "loss", "gradient_norm"]:
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError("nonfinite or negative training metric")
        if not math.isclose(
            row["feature_loss"] + row["l2_penalty"],
            row["loss"],
            rel_tol=1e-5,
            abs_tol=1e-7,
        ):
            raise ValueError("training objective does not match its components")
        prior_tokens = row["tokens_seen"]
    if prior_tokens != complete["tokens_seen"]:
        raise ValueError("final token exposure differs from training log")
    validation = [json.loads(line) for line in validation_path.read_text().splitlines()]
    if [row["step"] for row in validation] != [
        0,
        *sorted(set(trial["checkpoint_steps"]) | {steps}),
    ]:
        raise ValueError("validation log does not cover every declared checkpoint")
    token_counts = {}
    domain_coverage = {}
    for row in validation:
        if row["regularization_included"] is not False:
            raise ValueError("validation must exclude training regularization")
        for split, count in [
            ("train", min(n, trial.get("diagnostic_train_records", 32))),
            ("validation", trial["validation_records"]),
        ]:
            group = row["groups"][split]
            if group["records"] != count or group["tokens"] < count:
                raise ValueError("validation diagnostic coverage changed")
            token_counts.setdefault(split, group["tokens"])
            if token_counts[split] != group["tokens"]:
                raise ValueError("validation token coverage changed across checkpoints")
            for key in [
                "objective",
                "relative_mse",
                "cosine_error",
                "relative_norm_error",
            ]:
                if not math.isfinite(group[key]) or group[key] < -1e-6:
                    raise ValueError("invalid fitting validation metric")
            if trial.get("report_domain_diagnostics"):
                domains = group.get("by_domain")
                if not isinstance(domains, dict) or not domains:
                    raise ValueError("missing per-domain validation diagnostics")
                required = trial.get("validation_required_domains")
                if split == "validation" and required and set(domains) != set(required):
                    raise ValueError(
                        "validation domain coverage differs from declaration"
                    )
                coverage = {}
                for name, values in domains.items():
                    if (
                        not isinstance(name, str)
                        or not name
                        or not isinstance(values["records"], int)
                        or values["records"] < 1
                        or not isinstance(values["tokens"], int)
                        or values["tokens"] < values["records"]
                    ):
                        raise ValueError("invalid domain record/token coverage")
                    coverage[name] = (values["records"], values["tokens"])
                domain_coverage.setdefault(split, coverage)
                if domain_coverage[split] != coverage:
                    raise ValueError("domain coverage changed across checkpoints")
                for key in ["records", "tokens"]:
                    if sum(values[key] for values in domains.values()) != group[key]:
                        raise ValueError(
                            "domain counts do not sum to aggregate coverage"
                        )
                for key in [
                    "objective",
                    "relative_mse",
                    "cosine_error",
                    "relative_norm_error",
                ]:
                    if any(
                        not math.isfinite(values[key]) or values[key] < -1e-6
                        for values in domains.values()
                    ):
                        raise ValueError("invalid per-domain validation metric")
                    weighted = (
                        sum(
                            values[key] * values["records"]
                            for values in domains.values()
                        )
                        / count
                    )
                    if not math.isclose(
                        weighted, group[key], rel_tol=1e-6, abs_tol=1e-7
                    ):
                        raise ValueError(
                            "domain metrics do not reproduce record-weighted aggregate"
                        )
    metadata = complete["cache_extraction_metadata"]
    input_width = metadata["target_hidden_size"] * len(metadata["target_layer_ids"])
    output_width = metadata["draft_hidden_size"]
    parameters = (
        input_width * output_width
        if trial["architecture"] == "dense"
        else trial["width"] * (input_width + output_width)
    )
    if complete["parameters"] != parameters:
        raise ValueError(
            "reported parameter count differs from actual interface dimensions"
        )
    timing_keys = [
        "training_update_seconds",
        "loop_seconds",
        "initial_validation_seconds",
        "validation_seconds",
        "checkpoint_export_seconds",
        "worker_total_seconds",
        "setup_seconds",
        "input_io_seconds",
    ]
    timings = {key: complete[key] for key in timing_keys}
    if any(not math.isfinite(v) or v < 0 for v in timings.values()):
        raise ValueError("invalid fitting cost")
    if timings["worker_total_seconds"] < timings["loop_seconds"]:
        raise ValueError("worker timing excludes part of its own fitting loop")
    return {
        "trial": trial,
        "parameters": parameters,
        "input_width": input_width,
        "output_width": output_width,
        **expected_counts,
        "tokens_seen": prior_tokens,
        "train_objective": validation[-1]["groups"]["train"]["objective"],
        "validation_objective": validation[-1]["groups"]["validation"]["objective"],
        "best_validation_step": min(
            validation,
            key=lambda r: (r["groups"]["validation"]["objective"], r["step"]),
        )["step"],
        "validation_trajectory": validation,
        "timing": timings,
        "peak_gpu_bytes": complete["peak_gpu_bytes"],
        "source_sha256": {
            p.name: digest(p)
            for p in [complete_path, validation_path, training_path, trial_path]
        },
        "scope": "Fresh fitting log and metadata consistency. Checkpoint hashes are separately bound by the GPU batch gate; this local audit does not reload remote checkpoint tensors. Feature-validation minima are descriptive, not decoding selection or task-quality evidence.",
    }
