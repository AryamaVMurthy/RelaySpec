"""Native draft construction and correctly shifted block-prediction objectives."""
import copy
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


TAPS = [1, 9, 17, 25, 33]


class ProgressHead(nn.Module):
    """Zero-initialized residual correction of native draft output features."""
    def __init__(self, width, rank, positions=None):
        super().__init__()
        self.positions = positions
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden):
        if self.positions is not None:
            selected = hidden[..., self.positions, :]
            result = hidden.clone()
            result[..., self.positions, :] = selected+self.up(self.down(selected))
            return result
        return hidden+self.up(self.down(hidden))


def progress_loss(logits, labels, valid, native_tokens, error_weight=1., kind="ce", native_logits=None, margin=.1, preserve_weight=1., soft_temperature=.1):
    """Supervise accepted proposals and their first rejection, never later labels."""
    losses = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="none").reshape(labels.shape)
    errors = valid & (native_tokens != labels)
    weights = valid.float()*(1+(error_weight-1)*errors.float())
    loss = ((losses*weights).sum(-1)/weights.sum(-1).clamp_min(1)).mean()
    if kind == "margin_kl":
        if native_logits is None:
            raise ValueError("Native distributions are needed for preservation")
        top = logits.float().topk(2, dim=-1)
        competitor = torch.where(top.indices[..., 0] == labels, top.values[..., 1], top.values[..., 0])
        correct = logits.float().gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        margin_loss = (F.relu(competitor-correct+margin)*errors).sum(-1)/errors.sum(-1).clamp_min(1)
        logq = F.log_softmax(native_logits.detach().float(), dim=-1)
        logp = F.log_softmax(logits.float(), dim=-1)
        kl = (logq.exp()*(logq-logp)).sum(-1)
        keep = valid & ~errors
        preservation = (kl*keep).sum(-1)/keep.sum(-1).clamp_min(1)
        loss = (margin_loss+preserve_weight*preservation).mean()
    elif kind == "soft_progress":
        top = logits.float().topk(2, dim=-1)
        competitor = torch.where(top.indices[..., 0] == labels, top.values[..., 1], top.values[..., 0])
        correct = logits.float().gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        soft_match = torch.sigmoid((correct-competitor)/soft_temperature)*valid
        loss = -soft_match.cumprod(-1).sum(-1).mean()
    elif kind != "ce":
        raise ValueError("Unknown progress objective")
    predictions = logits.argmax(-1)
    return loss, {"valid_positions": int(valid.sum()), "correct_valid": int(((predictions == labels)&valid).sum()), "first_errors": int(errors.sum()), "first_errors_repaired": int(((predictions == labels)&errors).sum()), "preserved": int(((predictions == labels)&valid&~errors).sum()), "originally_correct": int((valid&~errors).sum())}


def stop_aware_prefix_mask(valid, native_tokens, stop_ids):
    """EOS itself can be a label, but a position conditioned after EOS cannot."""
    stops = torch.isin(native_tokens, torch.tensor(stop_ids, device=native_tokens.device))
    active = torch.cat([torch.ones_like(valid[:, :1]), ~stops.cummax(-1).values[:, :-1]], dim=-1)
    return valid & active


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


def block_batch(records, anchors, taps, block_size=16):
    """Right-pad prefix keys; each query retains its actual sequence position."""
    if not records or len(records) != len(anchors):
        raise ValueError("Expected one anchor per nonempty batch record")
    views = [block_view(r, a, taps, block_size) for r, a in zip(records, anchors)]
    context_length = max(anchors)
    context = views[0]["context"].new_zeros(len(records), context_length, views[0]["context"].shape[-1])
    positions = torch.zeros(len(records), context_length+block_size, dtype=torch.long)
    mask = torch.zeros(len(records), 1, block_size, context_length+block_size, dtype=torch.bool)
    for i, (view, anchor) in enumerate(zip(views, anchors)):
        context[i, :anchor] = view["context"][0]
        positions[i, :anchor] = torch.arange(anchor)
        positions[i, context_length:] = torch.arange(anchor, anchor+block_size)
        mask[i, :, :, :anchor] = True
        mask[i, :, :, context_length:] = True
    return {"context": context, "positions": positions, "attention_mask": mask,
            "anchor_token": torch.cat([v["anchor_token"] for v in views]),
            "labels": torch.stack([v["labels"] for v in views]),
            "teacher_hidden": torch.stack([v["teacher_hidden"] for v in views])}


def conditioned_noise(view, mask_token_id, prefix=0):
    """Reveal a training prefix; its copied labels must be excluded from loss."""
    if not 0 <= prefix < view["labels"].shape[-1]:
        raise ValueError("Conditioned prefix leaves no prediction target")
    labels = view["labels"]
    noise = torch.full((len(labels), labels.shape[-1]+1), mask_token_id, dtype=labels.dtype, device=labels.device)
    noise[:, 0] = view["anchor_token"]
    noise[:, 1:prefix+1] = labels[:, :prefix]
    return noise


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
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none").reshape(labels.shape)
    hard = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), teacher.argmax(-1).reshape(-1), reduction="none").reshape(labels.shape)
    losses = {"kl": kl, "ce": ce, "hard_target": hard, "mixed": .5*(kl+ce)}
    if kind not in losses:
        raise ValueError("Unknown token objective")
    match = logits.argmax(-1) == teacher.argmax(-1)
    return (losses[kind]*weights).sum(-1).mean(), {
        "kl": float((kl*weights).sum(-1).mean().detach()),
        "data_ce": float((ce*weights).sum(-1).mean().detach()),
        "token_agreement": float(match.float().mean()),
        "teacher_forced_prefix_matches": float(match.long().cumprod(-1).sum(-1).float().mean()),
    }
