"""Drafter-internal LoRA with a merge-equivalent BF16 training forward."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class DraftLoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        assert rank > 0
        self.base = base.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.rank = rank

    def weight_for_export(self):
        # alpha=rank. Explicit FP32 accumulation then deployment BF16 rounding.
        with torch.autocast(self.base.weight.device.type, enabled=False):
            return (self.base.weight.float() + self.B @ self.A).to(self.base.weight.dtype)

    def forward(self, x):
        return F.linear(x, self.weight_for_export(), self.base.bias)


def inject(draft, rank):
    names = []
    suffixes = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
    for name, module in list(draft.named_modules()):
        if isinstance(module, nn.Linear) and name.rsplit('.', 1)[-1] in suffixes:
            parent, _, child = name.rpartition('.')
            setattr(draft.get_submodule(parent) if parent else draft, child, DraftLoRALinear(module, rank))
            names.append(name)
    assert names, 'No drafter transformer projections matched'
    return names


def parameters_by_name(draft):
    return {name: p for name, p in draft.named_parameters() if p.requires_grad}


@torch.no_grad()
def merged_state(draft):
    result = {}
    for name, value in draft.state_dict().items():
        if name.endswith(('.A', '.B')):
            continue
        result[name.replace('.base.', '.')] = value.detach().cpu().contiguous()
    for name, module in draft.named_modules():
        if isinstance(module, DraftLoRALinear):
            result[name + '.weight'] = module.weight_for_export().detach().cpu().contiguous()
    return result


@torch.no_grad()
def export_gate(draft, embedding, mapper, example):
    """Check complete logits after reloading merged transformer weights."""
    from .train_pilot import block_for, logits
    from .blocks import collate_blocks
    batch = collate_blocks([block_for(example, len(example[0]['prompt_token_ids']), draft)], 'cuda')
    state = merged_state(draft)
    replacements = []
    with torch.autocast('cuda', dtype=torch.bfloat16):
        expected = logits(draft, embedding, mapper, batch)
        for name, module in list(draft.named_modules()):
            if not isinstance(module, DraftLoRALinear):
                continue
            parent_name, _, child = name.rpartition('.')
            parent = draft.get_submodule(parent_name) if parent_name else draft
            plain = nn.Linear(module.base.in_features, module.base.out_features,
                              bias=module.base.bias is not None, device='cuda', dtype=torch.bfloat16)
            plain.weight.copy_(state[name + '.weight'])
            if plain.bias is not None:
                plain.bias.copy_(state[name + '.bias'])
            setattr(parent, child, plain)
            replacements.append((parent, child, module))
        try:
            actual = logits(draft, embedding, mapper, batch)
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        finally:
            for parent, child, module in replacements:
                setattr(parent, child, module)
    return state, {'merged_transformer_logits_exact': True}
