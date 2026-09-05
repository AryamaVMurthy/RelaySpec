import json

import pytest
import torch

from relayspec.feature_cache import (
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
