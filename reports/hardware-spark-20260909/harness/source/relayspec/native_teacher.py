"""Preserve the released drafter's exact post-fusion interface convention."""

import torch


def export_native_teacher(draft, config):
    family = config["proposer"]["family"]
    if family not in {"dflash", "eagle3"}:
        raise ValueError("Unsupported native teacher family")
    result = dict(
        family=family,
        weight=draft.fc.weight.detach().cpu(),
        target_layer_ids=list(draft.target_layer_ids),
        target=config["target"],
        proposer=config["proposer"],
    )
    if family == "dflash":
        result.update(
            norm_weight=draft.hidden_norm.weight.detach().cpu(),
            eps=draft.hidden_norm.variance_epsilon,
        )
    return result


def restore_native_norm(teacher, device):
    family = teacher.get("family", "dflash")
    if family == "eagle3":
        if "norm_weight" in teacher:
            raise ValueError("EAGLE post-fusion interface has no external output norm")
        return torch.nn.Identity().to(device)
    if family != "dflash":
        raise ValueError("Unsupported native teacher family")
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    norm = Qwen3RMSNorm(teacher["weight"].shape[0], eps=teacher["eps"])
    norm.load_state_dict({"weight": teacher["norm_weight"]}, strict=True)
    return (
        norm.to(device=device, dtype=teacher["weight"].dtype)
        .requires_grad_(False)
        .eval()
    )
