import json

import pytest
import torch

from relayspec.feature_cache import (
    MappedFeatureReader,
    finalize_cache,
    load_cached_record,
    pad_feature_batch,
    record_weighted_loss,
    write_cache_shard,
)
from relayspec.losses import interface_alignment_loss


def test_cached_roundtrip_coverage_and_corruption_detection(tmp_path):
    groups = {"train": [{"i": i} for i in range(8)], "validation": [{"i": 9}]}

    def extract(row):
        n = row["i"] + 1
        return (
            torch.full((1, n, 3), 2.0, dtype=torch.bfloat16),
            torch.ones(1, n, 2),
            torch.arange(n)[None],
        )

    for rank in range(4):
        write_cache_shard(tmp_path, groups, rank=rank, world_size=4, extract=extract)
    index = finalize_cache(
        tmp_path, counts={"train": 8, "validation": 1}, world_size=4, metadata={}
    )
    x, y = load_cached_record(tmp_path, index["entries"][0])
    assert torch.equal(x, extract({"i": 0})[0]) and x.dtype == torch.bfloat16
    path = tmp_path / index["entries"][0]["file"]
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        load_cached_record(tmp_path, index["entries"][0])
    assert (
        json.loads((tmp_path / "cache-index.json").read_text())["counts"]["train"] == 8
    )


def test_mapped_reader_preserves_tensors_and_rejects_changed_inputs(tmp_path):
    def extract(row):
        return (
            torch.arange(12, dtype=torch.bfloat16).reshape(1, 4, 3),
            torch.ones(1, 4, 2),
            torch.arange(4)[None],
        )

    entries = write_cache_shard(
        tmp_path, {"train": [{"i": 0}]}, rank=0, world_size=1, extract=extract
    )
    entry = entries[0]
    reader = MappedFeatureReader(tmp_path)
    baseline = load_cached_record(tmp_path, entry)
    first = reader(entry)
    second = reader(entry)
    assert first is second
    assert all(torch.equal(a, b) for a, b in zip(first, baseline, strict=True))
    with pytest.raises(ValueError, match="changed"):
        reader({**entry, "sha256": "0" * 64})
    (tmp_path / entry["file"]).touch()
    with pytest.raises(ValueError, match="changed"):
        reader(entry)
    with pytest.raises(ValueError, match="hash"):
        MappedFeatureReader(tmp_path)({**entry, "sha256": "0" * 64})


@pytest.mark.parametrize("objective", ["relative_interface_mse", "raw_mse"])
def test_masked_batch_matches_mean_per_record_losses_and_gradients(objective):
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2, bias=False)
    examples = [(torch.randn(1, n, 3), torch.randn(1, n, 2)) for n in [2, 7, 3, 11]]
    reference = torch.stack(
        [
            interface_alignment_loss(model(x), y, objective=objective)
            for x, y in examples
        ]
    ).mean()
    grad = torch.autograd.grad(reference, model.weight)[0]
    x, y, mask = pad_feature_batch(examples, torch.device("cpu"))
    loss = record_weighted_loss(model(x), y, mask, objective=objective)
    batched_grad = torch.autograd.grad(loss, model.weight)[0]
    torch.testing.assert_close(reference, loss)
    torch.testing.assert_close(grad, batched_grad)
    altered = model(x).detach()
    altered[~mask] = 1e9
    torch.testing.assert_close(
        record_weighted_loss(altered, y, mask, objective=objective), loss.detach()
    )
