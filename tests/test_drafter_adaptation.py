import copy

import pytest
import torch
from torch import nn

from relayspec.drafter_adaptation import (
    apply_merged_lora,
    export_merged_lora,
    frozen_base_digest,
    restore_adaptation_state,
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


@pytest.mark.parametrize("use_lora", [False, True])
def test_adaptation_resume_preserves_adam_trajectory_and_rejects_changed_data(use_lora):
    torch.manual_seed(37)
    base = nn.Sequential(nn.Linear(4, 4))
    if use_lora:
        base.requires_grad_(False)
        inject_lora(base, ("0",), 2, 4)
    uninterrupted = copy.deepcopy(base)
    partial = copy.deepcopy(base)

    def optimizer(model):
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.002,
            weight_decay=0.0,
        )

    inputs = torch.randn(4, 3, 4)
    targets = torch.randn(4, 3, 4)

    def update(model, opt, step):
        opt.zero_grad(set_to_none=True)
        (model(inputs[step]) - targets[step]).square().mean().backward()
        opt.step()

    full_optimizer = optimizer(uninterrupted)
    partial_optimizer = optimizer(partial)
    for step in range(4):
        update(uninterrupted, full_optimizer, step)
    for step in range(2):
        update(partial, partial_optimizer, step)
    state = copy.deepcopy(
        {
            "format": "relayspec-adaptation-training-state-v1",
            "config_sha256": "parent",
            "initial_mapper_sha256": "mapper",
            "feature_cache_index_sha256": "cache",
            "ordered_record_files": ["a", "b"],
            "lora_rank": 2 if use_lora else 0,
            "updates": 2,
            "trainable": {
                name: p.detach().clone()
                for name, p in partial.named_parameters()
                if p.requires_grad
            },
            "optimizer": partial_optimizer.state_dict(),
            "cpu_rng": torch.get_rng_state(),
        }
    )
    resumed = copy.deepcopy(base)
    resumed_optimizer = optimizer(resumed)
    args = dict(
        parent_config_sha256="parent",
        mapper_sha256="mapper",
        feature_cache_sha256="cache",
        ordered_record_files=["a", "b"],
        lora_rank=state["lora_rank"],
    )
    before = {name: value.clone() for name, value in resumed.state_dict().items()}
    with pytest.raises(ValueError, match="provenance"):
        restore_adaptation_state(
            resumed,
            resumed_optimizer,
            state,
            **{**args, "ordered_record_files": ["b", "a"]},
        )
    assert all(
        torch.equal(before[name], value) for name, value in resumed.state_dict().items()
    )
    invalid = copy.deepcopy(state)
    next(iter(invalid["optimizer"]["state"].values()))["exp_avg"].fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        restore_adaptation_state(resumed, resumed_optimizer, invalid, **args)
    assert all(
        torch.equal(before[name], value) for name, value in resumed.state_dict().items()
    )
    start = restore_adaptation_state(resumed, resumed_optimizer, state, **args)
    for step in range(start, 4):
        update(resumed, resumed_optimizer, step)
    assert all(
        torch.equal(value, uninterrupted.state_dict()[name])
        for name, value in resumed.state_dict().items()
    )
