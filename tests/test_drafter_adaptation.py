import copy

import pytest
import torch
from torch import nn

from relayspec.drafter_adaptation import (
    apply_merged_lora,
    export_merged_lora,
    frozen_base_digest,
)
from relayspec.lora import inject_lora, merge_lora_into_base


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_export_preserves_training_model_and_reload_matches_merged_reference(dtype):
    torch.manual_seed(17)
    base = nn.Sequential(nn.Linear(8, 12), nn.GELU(), nn.Linear(12, 8)).to(dtype)
    base.requires_grad_(False)
    fingerprint = frozen_base_digest(base)
    trained = copy.deepcopy(base)
    params = inject_lora(trained, ("0", "2"), 2, 4.0)
    assert frozen_base_digest(trained) == fingerprint
    x = torch.randn(4, 8).to(dtype)
    assert torch.equal(base(x), trained(x))
    with torch.no_grad():
        for parameter in params:
            parameter.add_(0.04)
    before = {name: value.clone() for name, value in trained.state_dict().items()}
    payload = export_merged_lora(
        trained,
        expected_base_digest=fingerprint,
        proposer={"id": "test"},
        mapper_sha256="map",
    )
    assert all(torch.equal(before[k], v) for k, v in trained.state_dict().items())
    merged = copy.deepcopy(trained)
    merge_lora_into_base(merged)
    reloaded = copy.deepcopy(base)
    apply_merged_lora(reloaded, payload, proposer={"id": "test"}, mapper_sha256="map")
    assert torch.equal(reloaded(x), merged(x))
    assert not torch.equal(reloaded(x), base(x))
    assert frozen_base_digest(trained) == fingerprint


def test_wrong_base_or_mapper_is_rejected_before_weight_changes():
    model = nn.Sequential(nn.Linear(4, 4)).requires_grad_(False)
    fingerprint = frozen_base_digest(model)
    adapted = copy.deepcopy(model)
    inject_lora(adapted, ("0",), 2, 2)
    payload = export_merged_lora(
        adapted,
        expected_base_digest=fingerprint,
        proposer={"id": "test"},
        mapper_sha256="map",
    )
    before = model[0].weight.clone()
    with pytest.raises(ValueError, match="provenance"):
        apply_merged_lora(
            model, payload, proposer={"id": "test"}, mapper_sha256="other"
        )
    assert torch.equal(before, model[0].weight)
    with torch.no_grad():
        adapted[0].base.weight.add_(1)
    with pytest.raises(ValueError, match="weights changed"):
        export_merged_lora(
            adapted,
            expected_base_digest=fingerprint,
            proposer={"id": "test"},
            mapper_sha256="map",
        )
