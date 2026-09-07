from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from relayspec.native_teacher import export_native_teacher, restore_native_norm


@pytest.mark.parametrize("family", ["dflash", "eagle3"])
def test_native_interface_roundtrip_uses_family_specific_normalization(family):
    draft = SimpleNamespace(
        fc=torch.nn.Linear(6, 4, bias=False), target_layer_ids=[1, 3]
    )
    if family == "dflash":
        draft.hidden_norm = Qwen3RMSNorm(4, eps=1e-6)
        with torch.no_grad():
            draft.hidden_norm.weight.copy_(torch.tensor([0.7, 0.8, 1.1, 1.4]))
    config = {"proposer": {"family": family}, "target": {"id": "test"}}
    teacher = export_native_teacher(draft, config)
    x = torch.randn(2, 3, 6)
    expected = draft.fc(x)
    if family == "dflash":
        expected = draft.hidden_norm(expected)
    norm = restore_native_norm(teacher, "cpu")
    actual = norm(torch.nn.functional.linear(x, teacher["weight"]))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(not p.requires_grad for p in norm.parameters())
    if family == "eagle3":
        assert "norm_weight" not in teacher
        with pytest.raises(ValueError):
            restore_native_norm({**teacher, "norm_weight": torch.ones(4)}, "cpu")
