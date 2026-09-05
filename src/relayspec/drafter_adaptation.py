"""Auditable inherited-drafter LoRA export without mutating the training model."""

import hashlib

import torch
from torch import nn

from relayspec.lora import LoRALinear


def frozen_base_digest(model):
    """Hash canonical inherited parameters, excluding only explicit LoRA deltas."""
    digest = hashlib.sha256()
    lora_names = {
        f"{name}.{part}" if name else part
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
        for part in ("lora_a", "lora_b")
    }
    parameters = []
    for name, parameter in model.named_parameters():
        if name in lora_names:
            continue
        if parameter.requires_grad:
            raise ValueError(f"inherited drafter parameter is not frozen: {name}")
        canonical = name.replace(".base.", ".")
        if canonical.startswith("base.") and isinstance(model, LoRALinear):
            canonical = canonical[5:]
        parameters.append((canonical, parameter))
    for name, parameter in sorted(parameters):
        value = parameter.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}:".encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def export_merged_lora(model, *, expected_base_digest, proposer, mapper_sha256):
    """Export merged linear weights into a new payload; retain live optimizer state."""
    if frozen_base_digest(model) != expected_base_digest:
        raise ValueError("inherited drafter weights changed during adaptation")
    updates = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            value = module.merged_weight().detach().cpu().contiguous()
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite merged LoRA weight")
            updates[f"{name}.weight" if name else "weight"] = value
    if not updates:
        raise ValueError("drafter has no LoRA updates to export")
    return {
        "format": "relayspec-merged-drafter-lora-v1",
        "base_sha256": expected_base_digest,
        "proposer": proposer,
        "mapper_sha256": mapper_sha256,
        "weights": updates,
        "scope": "Merged BF16/FP32 inherited linear weights. Export is independent of "
        "the live training model. BF16 merge rounding may differ from an unmerged "
        "LoRA forward; validate reload against the merged reference.",
    }


@torch.no_grad()
def apply_merged_lora(model, payload, *, proposer, mapper_sha256):
    """Validate the whole update before modifying a freshly loaded plain drafter."""
    if (
        payload.get("format") != "relayspec-merged-drafter-lora-v1"
        or payload.get("proposer") != proposer
        or payload.get("mapper_sha256") != mapper_sha256
    ):
        raise ValueError("adaptation payload has different model/mapper provenance")
    if frozen_base_digest(model) != payload["base_sha256"]:
        raise ValueError("adaptation payload requires its exact inherited drafter")
    if not payload.get("weights"):
        raise ValueError("empty drafter update")
    assignments = []
    for name, value in payload["weights"].items():
        if not name.endswith(".weight") and name != "weight":
            raise ValueError("only linear weight updates are supported")
        module = (
            model.get_submodule(name.removesuffix(".weight"))
            if name != "weight"
            else model
        )
        if (
            type(module) is not nn.Linear
            or value.shape != module.weight.shape
            or value.dtype != module.weight.dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"incompatible drafter linear update: {name}")
        assignments.append((module.weight, value))
    for parameter, value in assignments:
        parameter.copy_(value.to(parameter.device))
