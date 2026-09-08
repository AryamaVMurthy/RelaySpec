"""Native draft construction and correctly shifted block-prediction objectives."""
import copy
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


TAPS = [1, 9, 17, 25, 33]


@torch.no_grad()
def cast_parameters(model, dtype):
    """Change trainable weight storage without rounding nonpersistent RoPE buffers."""
    for parameter in model.parameters():
        parameter.data = parameter.data.to(dtype=dtype)
        if parameter.grad is not None:
            parameter.grad.data = parameter.grad.data.to(dtype=dtype)
    return model


def token_ids(value):
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected one tokenized sequence")
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(v, int) for v in value):
        raise ValueError("Unexpected tokenizer output")
    return value


def compact_student(native, taps, layers):
    if not set(taps).issubset(TAPS) or not layers or len(set(layers)) != len(layers):
        raise ValueError("Invalid student architecture")
    student = copy.deepcopy(native)
    width = native.config.hidden_size
    columns = torch.cat([torch.arange(TAPS.index(t)*width, (TAPS.index(t)+1)*width, device=native.fc.weight.device) for t in taps])
    student.fc = nn.Linear(len(columns), width, bias=False, device=native.fc.weight.device, dtype=native.fc.weight.dtype)
    with torch.no_grad():
        student.fc.weight.copy_(native.fc.weight[:, columns])
    student.layers = nn.ModuleList([student.layers[i] for i in layers])
    for i, layer in enumerate(student.layers):
        layer.self_attn.layer_idx = i
    student.target_layer_ids = list(taps)
    student.config.num_hidden_layers = len(layers)
    student.config.dflash_config["target_layer_ids"] = list(taps)
    return student


def block_view(record, anchor, taps, block_size=16):
    """Inputs stop before the anchor; target logits are shifted by one token."""
    ids = record["input_ids"]
    if not 1 <= anchor <= len(ids)-block_size:
        raise ValueError("Invalid supervised anchor")
    context = torch.cat([record["taps"][TAPS.index(t), :anchor] for t in taps], dim=-1).unsqueeze(0)
    return {
        "context": context,
        "anchor_token": ids[anchor:anchor+1],
        "labels": ids[anchor+1:anchor+block_size],
        "teacher_hidden": record["final_hidden"][anchor:anchor+block_size-1],
        "positions": torch.arange(anchor+block_size).unsqueeze(0),
    }


def prediction_loss(student_logits, teacher_logits, labels, kind="kl", gamma=7.0, temperature=1.0):
    logits = student_logits.float()
    weights = torch.ones(logits.shape[-2], device=logits.device)
    if gamma:
        weights = torch.exp(-torch.arange(len(weights), device=logits.device)/gamma)
    weights = weights / weights.sum()
    teacher = teacher_logits.detach().float()
    logp = F.log_softmax(logits/temperature, dim=-1)
    logq = F.log_softmax(teacher/temperature, dim=-1)
    kl = (logq.exp()*(logq-logp)).sum(-1)*temperature**2
    ce = F.cross_entropy(logits, labels, reduction="none")
    hard = F.cross_entropy(logits, teacher.argmax(-1), reduction="none")
    losses = {"kl": kl, "ce": ce, "hard_target": hard, "mixed": .5*(kl+ce)}
    if kind not in losses:
        raise ValueError("Unknown token objective")
    match = logits.argmax(-1) == teacher.argmax(-1)
    return (losses[kind]*weights).sum(), {
        "kl": float((kl*weights).sum().detach()),
        "data_ce": float((ce*weights).sum().detach()),
        "token_agreement": float(match.float().mean()),
        "teacher_forced_prefix_matches": int(match.long().cumprod(-1).sum()),
    }
