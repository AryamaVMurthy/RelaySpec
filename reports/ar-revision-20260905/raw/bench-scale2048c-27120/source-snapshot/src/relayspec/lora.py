from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank additive delta.

    Standard LoRA (Hu et al., 2021): `y = W x + (alpha / r) * B (A x)`,
    with `W` frozen, `A` initialized from a scaled normal distribution and
    `B` initialized to zero so the wrapped layer is an exact no-op before
    any training happens. Implemented directly (no `peft` dependency) since
    only linear-layer wrapping is needed here, not the full adapter-config
    machinery `peft` provides.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_a = nn.Parameter(
            torch.randn(rank, base.in_features, device=device, dtype=torch.float32)
            * (1.0 / math.sqrt(base.in_features))
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, rank, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = (x.float() @ self.lora_a.T) @ self.lora_b.T
        return base_out + (self.scaling * delta).to(base_out.dtype)

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        delta = self.scaling * (self.lora_b.float() @ self.lora_a.float())
        return (self.base.weight.float() + delta).to(self.base.weight.dtype)


def inject_lora(model: nn.Module, target_suffixes: tuple[str, ...], rank: int, alpha: float) -> list[nn.Parameter]:
    """Wrap every linear submodule whose name ends in one of `target_suffixes`.

    Returns the trainable LoRA parameters. Everything else in `model`
    should already be frozen by the caller; this only adds new trainable
    parameters, it does not change any existing `requires_grad` state
    outside the wrapped layers.
    """
    trainable: list[nn.Parameter] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(suffix) for suffix in target_suffixes):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapped = LoRALinear(module, rank=rank, alpha=alpha)
        setattr(parent, child_name, wrapped)
        trainable.append(wrapped.lora_a)
        trainable.append(wrapped.lora_b)
    if not trainable:
        raise ValueError(f"no linear layers matched suffixes {target_suffixes}")
    return trainable


@torch.no_grad()
def merge_lora_into_base(model: nn.Module) -> None:
    """Replace every LoRALinear in `model` with a plain Linear holding the
    merged weight, so the result is an ordinary standalone checkpoint with
    no dependency on this module at load time."""
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        merged = nn.Linear(
            module.base.in_features,
            module.base.out_features,
            bias=module.base.bias is not None,
            device=module.base.weight.device,
            dtype=module.base.weight.dtype,
        )
        merged.weight.copy_(module.merged_weight())
        if module.base.bias is not None:
            merged.bias.copy_(module.base.bias)
        setattr(parent, child_name, merged)
