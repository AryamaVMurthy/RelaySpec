import json

import pytest

from relayspec.cached_fit_evidence import audit_fit_artifacts


@pytest.fixture
def fit(tmp_path):
    trial = {
        "name": "test-fit",
        "architecture": "dense",
        "steps": 2,
        "distinct_examples": 8,
        "checkpoint_steps": [1, 2],
        "validation_records": 4,
        "diagnostic_train_records": 8,
    }
    complete = {
        "status": "pass",
        "trial": trial,
        "start_step": 0,
        "steps": 2,
        "updates_this_job": 2,
        "record_presentations": 8,
        "epochs": 1,
        "distinct_records_seen": 8,
        "tokens_seen": 16,
        "feature_cache_index_sha256": "cache",
        "parameters": 6,
        "cache_extraction_metadata": {
            "target_hidden_size": 3,
            "target_layer_ids": [1],
            "draft_hidden_size": 2,
        },
        "training_update_seconds": 2,
        "loop_seconds": 4,
        "initial_validation_seconds": 1,
        "validation_seconds": 1,
        "checkpoint_export_seconds": 1,
        "worker_total_seconds": 6,
        "setup_seconds": 1,
        "input_io_seconds": 0.5,
        "peak_gpu_bytes": 100,
    }
    rows = [
        {
            "step": s,
            "record_presentations": s * 4,
            "distinct_records_seen": s * 4,
            "epochs": s / 2,
            "tokens_seen": s * 8,
            "feature_loss": 0.5,
            "l2_penalty": 0.0,
            "loss": 0.5,
            "gradient_norm": 0.3,
        }
        for s in [1, 2]
    ]
    points = [
        {
            "step": s,
            "regularization_included": False,
            "groups": {
                name: {
                    "records": count,
                    "tokens": 2 * count,
                    "objective": 0.5,
                    "relative_mse": 0.5,
                    "cosine_error": 0.4,
                    "relative_norm_error": 0.2,
                }
                for name, count in [("train", 8), ("validation", 4)]
            },
        }
        for s in [0, 1, 2]
    ]
    (tmp_path / "fit-complete.json").write_text(json.dumps(complete))
    (tmp_path / "trial.json").write_text(json.dumps(trial))
    (tmp_path / "training.jsonl").write_text("\n".join(map(json.dumps, rows)))
    (tmp_path / "validation.jsonl").write_text("\n".join(map(json.dumps, points)))
    return tmp_path, trial


def test_reconstruct_exposure_dimensions_and_validation(fit):
    folder, trial = fit
    result = audit_fit_artifacts(folder, trial, cache_sha256="cache")
    assert result["record_presentations"] == 8
    assert result["tokens_seen"] == 16
    assert result["parameters"] == 6
    assert result["best_validation_step"] == 0
    assert len(result["validation_trajectory"]) == 3
    assert len(result["source_sha256"]) == 4


@pytest.mark.parametrize(
    "damage",
    [
        "missing_update",
        "exposure",
        "nan",
        "tokens",
        "parameters",
        "validation",
        "cache",
    ],
)
def test_reject_incomplete_or_inconsistent_fit_evidence(fit, damage):
    folder, trial = fit
    path = folder / "training.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if damage == "missing_update":
        rows.pop(0)
    elif damage == "exposure":
        rows[0]["distinct_records_seen"] = 8
    elif damage == "nan":
        rows[0]["loss"] = float("nan")
    elif damage == "tokens":
        rows[-1]["tokens_seen"] = 17
    path.write_text("\n".join(map(json.dumps, rows)))
    if damage == "parameters":
        path = folder / "fit-complete.json"
        data = json.loads(path.read_text())
        data["parameters"] = 7
        path.write_text(json.dumps(data))
    if damage == "validation":
        path = folder / "validation.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]))
    with pytest.raises(ValueError):
        audit_fit_artifacts(
            folder, trial, cache_sha256="wrong" if damage == "cache" else "cache"
        )


@pytest.mark.parametrize(
    "damage", [None, "missing", "labels", "counts", "nan", "weighting", "drift"]
)
def test_composition_domains_reconstruct_fixed_record_weighted_metrics(fit, damage):
    folder, trial = fit
    trial.update(
        report_domain_diagnostics=True, validation_required_domains=["math", "general"]
    )
    path = folder / "fit-complete.json"
    complete = json.loads(path.read_text())
    complete["trial"] = trial
    path.write_text(json.dumps(complete))
    (folder / "trial.json").write_text(json.dumps(trial))
    path = folder / "validation.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        for group in row["groups"].values():
            total = group["records"]
            domains = {}
            # Unequal domain sizes/lengths distinguish record from token/domain weighting.
            for name, count, tokens, metric in [
                ("math", 1, 1, 0.2),
                ("general", total - 1, group["tokens"] - 1, 0.6),
            ]:
                domains[name] = {
                    "records": count,
                    "tokens": tokens,
                    **{
                        key: metric
                        for key in [
                            "objective",
                            "relative_mse",
                            "cosine_error",
                            "relative_norm_error",
                        ]
                    },
                }
            for key in [
                "objective",
                "relative_mse",
                "cosine_error",
                "relative_norm_error",
            ]:
                group[key] = (0.2 + 0.6 * (total - 1)) / total
            group["by_domain"] = domains
    group = rows[-1]["groups"]["validation"]
    if damage == "missing":
        group.pop("by_domain")
    elif damage == "labels":
        group["by_domain"]["other"] = group["by_domain"].pop("math")
    elif damage == "counts":
        group["by_domain"]["math"]["tokens"] += 1
    elif damage == "nan":
        group["by_domain"]["math"]["objective"] = float("nan")
    elif damage == "weighting":
        group["objective"] = 0.4
    elif damage == "drift":
        group["by_domain"]["math"]["tokens"] += 1
        group["by_domain"]["general"]["tokens"] -= 1
    path.write_text("\n".join(map(json.dumps, rows)))
    if damage is None:
        result = audit_fit_artifacts(folder, trial, cache_sha256="cache")
        assert result["validation_objective"] == pytest.approx(0.5)
    else:
        with pytest.raises(ValueError):
            audit_fit_artifacts(folder, trial, cache_sha256="cache")
