import pytest
import torch
from torch import nn

from relayspec.benchmarking import model_load_plan
from relayspec.unfitted_controls import UnfittedContext


def test_no_projection_uses_last_tap_and_has_no_parameters():
    control = UnfittedContext(target_width=4, draft_width=2, num_taps=2)
    features = torch.arange(8.0).reshape(1, 1, 8)
    assert control(features).tolist() == [[[4.0, 5.0]]]
    assert list(control.parameters()) == []
    with pytest.raises(ValueError):
        control(torch.zeros(1, 1, 4))


def test_original_fc_receives_each_tap_without_channel_mixing():
    fc = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        fc.weight.copy_(torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]))
    control = UnfittedContext(target_width=4, draft_width=2, num_taps=2, source_fc=fc)
    assert control(torch.arange(8.0).reshape(1, 1, 8)).tolist() == [[[4.0, 6.0]]]
    assert not any(p.requires_grad for p in control.parameters())
    plan = model_load_plan(["direct_slice"], unload_source_trunk=True)
    assert plan["load_source"] and plan["unload_source_trunk"]
    assert not plan["build_source_provider"]
