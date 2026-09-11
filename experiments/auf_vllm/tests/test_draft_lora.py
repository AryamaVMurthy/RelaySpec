import torch
from torch import nn
from experiments.auf_vllm.draft_lora import DraftLoRALinear, inject, merged_state


def test_frozen_base_nonzero_adaptation_and_merged_reload():
    torch.manual_seed(42)
    model=nn.ModuleDict({'q_proj':nn.Linear(7,5,bias=True),'fc':nn.Linear(7,5)})
    model.requires_grad_(False)
    original=model['q_proj'].weight.detach().clone()
    x=torch.randn(3,7)
    before=model['q_proj'](x).detach()
    assert inject(model,3)==['q_proj']
    layer=model['q_proj']
    torch.testing.assert_close(layer(x),before,rtol=0,atol=0)
    optimizer=torch.optim.SGD([layer.A,layer.B],lr=.1)
    for _ in range(2):
        optimizer.zero_grad();layer(x).square().mean().backward();optimizer.step()
    assert layer.A.grad.abs().sum()>0 and layer.B.grad.abs().sum()>0
    assert layer.base.weight.grad is None and not model['fc'].weight.requires_grad
    torch.testing.assert_close(layer.base.weight,original,rtol=0,atol=0)
    plain=nn.ModuleDict({'q_proj':nn.Linear(7,5,bias=True),'fc':nn.Linear(7,5)})
    plain.load_state_dict(merged_state(model),strict=True)
    torch.testing.assert_close(layer(x),plain['q_proj'](x),rtol=0,atol=0)
    assert not torch.equal(layer(x),before)
