import pytest
import torch

from relayspec.feature_cache import load_cached_record, write_cache_shard
from relayspec.fitting_validation import interface_diagnostics


class CountIdentity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, values):
        self.calls += 1
        return values


def test_domains_preserve_overall_record_weighting_without_extra_forwards():
    examples = [
        (torch.ones(1, 2, 3), torch.full((1, 2, 3), 2.0)),
        (torch.ones(1, 5, 3), torch.full((1, 5, 3), 4.0)),
    ]
    model = CountIdentity()
    kwargs = {"device": torch.device("cpu"), "objective": "relative_interface_mse"}
    baseline = interface_diagnostics(model, examples, **kwargs)
    model.calls = 0
    reported = interface_diagnostics(
        model, examples, domains=["math", "general_instruction"], **kwargs
    )
    assert model.calls == len(examples)
    assert {k: v for k, v in reported.items() if k != "by_domain"} == baseline
    groups = reported["by_domain"]
    assert groups["math"]["tokens"] == 2
    assert groups["general_instruction"]["tokens"] == 5
    assert all(g["records"] == 1 for g in groups.values())
    assert sum(g["objective"] for g in groups.values()) / 2 == pytest.approx(
        reported["objective"]
    )
    with pytest.raises(ValueError, match="domain label"):
        interface_diagnostics(model, examples, domains=["math"], **kwargs)


def test_cache_preserves_domain_metadata_and_exact_tensor_payload(tmp_path):
    x, y, ids = torch.ones(1, 2, 3), torch.ones(1, 2, 2), torch.tensor([[2, 3]])
    entries = write_cache_shard(
        tmp_path,
        {"train": [{"domain": "math"}]},
        rank=0,
        world_size=1,
        extract=lambda _: (x, y, ids),
    )
    assert entries[0]["domain"] == "math"
    restored = load_cached_record(tmp_path, entries[0])
    assert torch.equal(restored[0], x) and torch.equal(restored[1], y)
