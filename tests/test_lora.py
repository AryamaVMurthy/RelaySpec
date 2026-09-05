from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


def test_lora_linear_is_a_no_op_before_training() -> None:
    from relayspec.lora import LoRALinear

    base = nn.Linear(8, 4, bias=False)
    wrapped = LoRALinear(base, rank=2, alpha=4.0)
    x = torch.randn(3, 8)
    with torch.no_grad():
        assert torch.allclose(wrapped(x), base(x))


def test_lora_linear_base_weight_stays_frozen() -> None:
    from relayspec.lora import LoRALinear

    base = nn.Linear(8, 4, bias=False)
    wrapped = LoRALinear(base, rank=2, alpha=4.0)
    assert not wrapped.base.weight.requires_grad
    assert wrapped.lora_a.requires_grad
    assert wrapped.lora_b.requires_grad


def test_lora_linear_training_changes_output_only_through_lora_params() -> None:
    from relayspec.lora import LoRALinear

    torch.manual_seed(0)
    base = nn.Linear(8, 4, bias=False)
    wrapped = LoRALinear(base, rank=2, alpha=4.0)
    x = torch.randn(5, 8)
    target = torch.randn(5, 4)
    optimizer = torch.optim.SGD([wrapped.lora_a, wrapped.lora_b], lr=0.1)
    before = base.weight.clone()
    for _ in range(5):
        optimizer.zero_grad()
        loss = ((wrapped(x) - target) ** 2).mean()
        loss.backward()
        optimizer.step()
    assert torch.equal(base.weight, before)
    assert not torch.allclose(wrapped(x), base(x))


def test_merge_lora_into_base_reproduces_forward_pass() -> None:
    from relayspec.lora import LoRALinear, merge_lora_into_base

    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(6, 6, bias=False))
    wrapped = LoRALinear(model[0], rank=3, alpha=6.0)
    model[0] = wrapped
    with torch.no_grad():
        wrapped.lora_a.copy_(torch.randn_like(wrapped.lora_a))
        wrapped.lora_b.copy_(torch.randn_like(wrapped.lora_b))
    x = torch.randn(4, 6)
    with torch.no_grad():
        before_merge = model(x)
    merge_lora_into_base(model)
    assert isinstance(model[0], nn.Linear)
    with torch.no_grad():
        after_merge = model(x)
    assert torch.allclose(before_merge, after_merge, atol=1e-5)


def test_inject_lora_wraps_only_matching_suffixes() -> None:
    from relayspec.lora import LoRALinear, inject_lora

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(4, 4, bias=False)
            self.o_proj = nn.Linear(4, 4, bias=False)
            self.unrelated = nn.Linear(4, 4, bias=False)

    model = nn.Sequential(Block())
    params = inject_lora(model, target_suffixes=("q_proj", "o_proj"), rank=2, alpha=4.0)
    assert isinstance(model[0].q_proj, LoRALinear)
    assert isinstance(model[0].o_proj, LoRALinear)
    assert not isinstance(model[0].unrelated, LoRALinear)
    assert len(params) == 4  # lora_a and lora_b for each of the 2 wrapped layers


def test_inject_lora_raises_when_nothing_matches() -> None:
    from relayspec.lora import inject_lora

    model = nn.Sequential(nn.Linear(4, 4, bias=False))
    with pytest.raises(ValueError):
        inject_lora(model, target_suffixes=("does_not_exist",), rank=2, alpha=4.0)
