import importlib.util
from pathlib import Path

import torch

from relayspec.mapper_campaign import restore_mapper

spec = importlib.util.spec_from_file_location(
    "research_lane", Path(__file__).parents[1] / "scripts/run_research_lane.py"
)
lane = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lane)


def checkpoint(tmp_path):
    path = tmp_path / "base.pt"
    state = {
        "target_layer_ids": [1, 3],
        "relay": {"projection.weight": torch.arange(24).float().reshape(3, 8)},
    }
    torch.save(state, path)
    return path, state


def test_drop_is_column_group_intervention_without_changing_parent(tmp_path):
    path, state = checkpoint(tmp_path)
    variants = lane.prepare(
        {"checkpoint": str(path), "transform": "drop_taps"}, tmp_path
    )
    dropped = torch.load(variants["relay_drop1"], weights_only=True)
    assert torch.count_nonzero(dropped["relay"]["projection.weight"][:, :4]) == 0
    torch.testing.assert_close(
        dropped["relay"]["projection.weight"][:, 4:],
        state["relay"]["projection.weight"][:, 4:],
    )
    torch.testing.assert_close(
        torch.load(path, weights_only=True)["relay"]["projection.weight"],
        state["relay"]["projection.weight"],
    )


def test_svd_export_loads_as_deployed_factorized_mapper(tmp_path, monkeypatch):
    path, state = checkpoint(tmp_path)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    variants = lane.prepare(
        {"checkpoint": str(path), "transform": "svd", "ranks": [2]}, tmp_path
    )
    cp = torch.load(variants["relay_svd2"], weights_only=True)
    mapper, _ = restore_mapper(cp, target_hidden_size=4, draft_hidden_size=3, eps=1e-6)
    product = mapper.projection[1].weight @ mapper.projection[0].weight
    torch.testing.assert_close(
        product, state["relay"]["projection.weight"], atol=2e-5, rtol=1e-5
    )


def test_activation_compression_preserves_frequent_feature_direction(
    tmp_path, monkeypatch
):
    import json

    from relayspec import research_compression

    entries = [dict(split="train", index=i) for i in range(512)] + [
        dict(split="validation", index=i) for i in range(128)
    ]
    (tmp_path / "cache-index.json").write_text(
        json.dumps(
            dict(
                status="pass",
                metadata=dict(target_layer_ids=[1], rms_norm_eps=1e-6),
                entries=entries,
            )
        )
    )
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    x = torch.tensor([[0.01, 100.0], [-0.01, 100.0], [0.01, -100.0], [-0.01, -100.0]])
    monkeypatch.setattr(
        research_compression,
        "load_cached_record",
        lambda root, entry: (x, torch.empty(0)),
    )
    base = dict(
        target_layer_ids=[1],
        relay_architecture="scale_preserving_linear",
        relay={"projection.weight": torch.diag(torch.tensor([10.0, 1.0]))},
    )
    states, diagnostics = research_compression.activation_compression(
        base, tmp_path, [1]
    )
    compressed = states[1]["projection.1.weight"] @ states[1]["projection.0.weight"]
    # Weight SVD keeps the first axis, but almost all observed output energy is on the second.
    torch.testing.assert_close(compressed[1, 1], torch.tensor(1.0))
    assert diagnostics["heldout_relative_projection_mse"]["1"] < 0.00001
