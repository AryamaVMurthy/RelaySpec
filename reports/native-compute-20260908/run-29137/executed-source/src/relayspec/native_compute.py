"""Experimental changes to a native drafter; target verification is untouched."""
from contextlib import contextmanager

import torch
from torch import nn


class ZeroMLP(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class LinearMLP(nn.Module):
    def __init__(self, down, up, bias):
        super().__init__()
        self.register_buffer("down", down)
        self.register_buffer("up", up)
        self.register_buffer("bias", bias)

    def forward(self, x):
        return (x @ self.down) @ self.up + self.bias


def fit_linear_mlp(x, y, rank, ridge=0.01):
    """PCA input basis and ridge residual prediction; no target-model updates."""
    x, y = x.float(), y.float()
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    rank = min(rank, x.shape[0] - 1, x.shape[1])
    _, _, basis = torch.svd_lowrank(xc, q=rank, niter=3)
    z = xc @ basis
    gram = z.T @ z
    penalty = ridge * gram.trace() / rank
    coeff = torch.linalg.solve(gram + penalty * torch.eye(rank, device=x.device), z.T @ yc)
    bias = ym - (xm @ basis) @ coeff
    prediction = (x @ basis) @ coeff + bias
    loss = ((prediction - y).square().sum(-1) / y.square().sum(-1).clamp_min(1e-8)).mean()
    return LinearMLP(basis.to(torch.bfloat16), coeff.to(torch.bfloat16), bias.to(torch.bfloat16)), float(loss)


@contextmanager
def native_intervention(draft, spec, bridges=None):
    """Restore every mutation even when a candidate fails."""
    import dflash.model as official

    original_layers = draft.layers
    original_mlps = [layer.mlp for layer in original_layers]
    original_indices = [layer.self_attn.layer_idx for layer in original_layers]
    original_sdpa = official.ALL_ATTENTION_FUNCTIONS["sdpa"]
    try:
        kind = spec.get("kind", "native")
        selected = spec.get("layers", [])
        if any(i < 0 or i >= len(original_layers) for i in selected):
            raise ValueError("Invalid native draft-layer selection")
        if kind == "drop_layers":
            kept = [layer for i, layer in enumerate(original_layers) if i not in selected]
            if not kept:
                raise ValueError("Cannot remove every attention/cache layer")
            draft.layers = nn.ModuleList(kept)
            for i, layer in enumerate(kept):
                layer.self_attn.layer_idx = i
        elif kind in {"zero_mlp", "linear_mlp"}:
            for i in selected:
                draft.layers[i].mlp = ZeroMLP() if kind == "zero_mlp" else bridges[(i, spec["rank"])]
        elif kind == "window":
            window = int(spec["window"])
            if window < draft.block_size:
                raise ValueError("Window must include the full proposal block")
            draft_attention = {layer.self_attn for layer in original_layers}

            def bounded_attention(module, query, key, value, attention_mask, **kwargs):
                if module in draft_attention and key.shape[-2] > window + 4:
                    # Absolute RoPE positions are already applied. Keep four sinks
                    # plus the recent window. Full cached K/V remain allocated.
                    indices = torch.cat((torch.arange(4, device=key.device), torch.arange(key.shape[-2] - window, key.shape[-2], device=key.device)))
                    key = key.index_select(-2, indices)
                    value = value.index_select(-2, indices)
                    if attention_mask is not None:
                        attention_mask = attention_mask.index_select(-1, indices)
                return original_sdpa(module, query, key, value, attention_mask, **kwargs)

            official.ALL_ATTENTION_FUNCTIONS["sdpa"] = bounded_attention
        elif kind != "native":
            raise ValueError(f"Unsupported native intervention: {kind}")
        yield
    finally:
        official.ALL_ATTENTION_FUNCTIONS["sdpa"] = original_sdpa
        draft.layers = original_layers
        for layer, mlp, index in zip(original_layers, original_mlps, original_indices):
            layer.mlp = mlp
            layer.self_attn.layer_idx = index
