import pytest

from relayspec.scaling_matrix import (
    four_gpu_batches,
    primary_matrix,
    regularization_matrix,
    trial,
    validate_cache_access_gate,
    validate_trial,
)


def test_crossed_coverage_and_matched_parameter_counts():
    cells = primary_matrix()
    assert len(cells) == 51
    assert len([t for t in cells if t["study"] == "core"]) == 35
    for n in [128, 512, 2048, 8192, 32768]:
        selected = [
            t for t in cells if t["distinct_examples"] == n and t["study"] == "core"
        ]
        assert len(selected) == 7
        assert {t["width"] for t in selected if t["architecture"] == "mlp"} == {
            128,
            512,
            2048,
        }
        assert {t["width"] for t in selected if t["architecture"] == "factorized"} == {
            128,
            512,
            2048,
        }


def test_budget_panels_do_not_inflate_distinct_data():
    for t in primary_matrix():
        validate_trial(t)
        n = t["distinct_examples"]
        fixed = t["budget_panels"]["fixed_exposure"]
        passes = t["budget_panels"]["fixed_passes"]
        assert fixed["presentations"] == 32768
        assert passes["presentations"] == 4 * n
        assert min(n, 4 * fixed["checkpoint_step"]) == n
        assert max(fixed["checkpoint_step"], passes["checkpoint_step"]) == t["steps"]
        assert n // 4 in t["checkpoint_steps"]
        assert n // 2 in t["checkpoint_steps"]
        assert 3 * n // 4 in t["checkpoint_steps"]
        assert n in t["checkpoint_steps"]


def test_batching_preserves_every_unique_cell_and_regularization_arm():
    cells = primary_matrix() + [trial("dense", None, 128, seed=1730)]
    batches = four_gpu_batches(cells)
    assert len(batches) == 13
    assert {t["name"] for b in batches for t in b} == {t["name"] for t in cells}
    regularized = regularization_matrix()
    assert len(regularized) == 42
    for t in regularized:
        validate_trial(t)
        assert bool(t["l2_weight"]) != bool(t["weight_decay"])
    with pytest.raises(ValueError, match="unique"):
        four_gpu_batches([cells[0]] * 4)
    invalid = trial("dense", None, 32768)
    invalid["steps"] = 1024
    with pytest.raises(ValueError, match="checkpoint"):
        validate_trial(invalid)


def test_optimized_access_rejects_missing_wrong_cache_or_inexact_gate():
    t = trial("dense", None, 512)
    validate_cache_access_gate([t], None, "cache-a")
    t.update(cache_backend="mmap", device_cache=True)
    with pytest.raises(ValueError, match="successful pilot"):
        validate_cache_access_gate([t], None, "cache-a")
    gate = {"status": "pass", "feature_cache_index_sha256": "cache-b"}
    with pytest.raises(ValueError, match="different feature cache"):
        validate_cache_access_gate([t], gate, "cache-a")
    gate["feature_cache_index_sha256"] = "cache-a"
    with pytest.raises(ValueError, match="exact fitted-state"):
        validate_cache_access_gate([t], gate, "cache-a")
    gate["pairs"] = [
        {"candidate": name, "weights_optimizer_and_loss_trajectory_bit_identical": True}
        for name in ["stream-mmap", "dense-device"]
    ]
    validate_cache_access_gate([t], gate, "cache-a")
    gate["pairs"][1]["weights_optimizer_and_loss_trajectory_bit_identical"] = False
    with pytest.raises(ValueError, match="exact fitted-state"):
        validate_cache_access_gate([t], gate, "cache-a")
