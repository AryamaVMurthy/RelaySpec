"""Independent DFlash blocks: full clean prefix, one anchor, then masks."""
from dataclasses import dataclass
import torch


@dataclass
class Block:
    context: torch.Tensor
    noise_ids: torch.Tensor
    labels: torch.Tensor
    valid: torch.Tensor
    anchor: int


def make_block(features, ids, anchor, block_size, mask_id, eos_ids=()):
    if features.ndim != 2 or len(features) != len(ids):
        raise ValueError("Dense features must align with every sequence token")
    if not 0 < anchor < len(ids)-1 or block_size < 2:
        raise ValueError("Anchor needs a nonempty prefix and at least one label")
    if any(token in eos_ids for token in ids[anchor:anchor+1]):
        raise ValueError("Cannot draft after an EOS anchor")
    noise = torch.full((block_size,), mask_id, dtype=torch.long)
    noise[0] = ids[anchor]
    labels = torch.full((block_size,), -100, dtype=torch.long)
    valid = torch.zeros(block_size, dtype=torch.bool)
    for j, token in enumerate(ids[anchor+1:anchor+block_size], start=1):
        labels[j] = token
        valid[j] = True
        if token in eos_ids:
            break
    # No feature at or after the clean anchor enters conditioning.
    return Block(features[:anchor], noise, labels, valid, anchor)


def collate_blocks(blocks, device):
    if not blocks:
        raise ValueError("Empty block batch")
    n, size = len(blocks), len(blocks[0].noise_ids)
    width = blocks[0].context.shape[-1]
    longest = max(b.anchor for b in blocks)
    context = torch.zeros(n, longest, width, dtype=blocks[0].context.dtype, device=device)
    positions = torch.zeros(n, longest+size, dtype=torch.long, device=device)
    allowed = torch.zeros(n, 1, size, longest+size, dtype=torch.bool, device=device)
    for index, block in enumerate(blocks):
        assert len(block.noise_ids) == size
        context[index, :block.anchor] = block.context.to(device)
        positions[index, :block.anchor] = torch.arange(block.anchor, device=device)
        positions[index, longest:] = torch.arange(block.anchor, block.anchor+size, device=device)
        allowed[index, :, :, :block.anchor] = True
        allowed[index, :, :, longest:] = True
    return {"context": context, "position_ids": positions, "attention_mask": allowed,
            "noise_ids": torch.stack([b.noise_ids for b in blocks]).to(device),
            "labels": torch.stack([b.labels for b in blocks]).to(device),
            "valid": torch.stack([b.valid for b in blocks]).to(device)}


def conditioned_forward(draft, context, noise, position_ids, attention_mask):
    """Official frozen layer path, retaining autograd into mapped context."""
    rotary = draft.rotary_emb(noise, position_ids)
    hidden = noise
    for layer in draft.layers:
        hidden = layer(hidden_states=hidden, target_hidden=context,
                       attention_mask=attention_mask, position_ids=position_ids,
                       position_embeddings=rotary, use_cache=False, is_causal=False)
    return draft.norm(hidden)
