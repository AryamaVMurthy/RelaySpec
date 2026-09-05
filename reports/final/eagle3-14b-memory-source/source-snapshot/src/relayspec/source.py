from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from relayspec.cache import DualCacheState


def causal_mask(
    *,
    batch_size: int,
    query_length: int,
    position_offset: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    total_length = position_offset + query_length
    query_positions = position_offset + torch.arange(query_length, device=device)
    key_positions = torch.arange(total_length, device=device)
    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    mask = torch.zeros((query_length, total_length), dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)


@dataclass
class SourceTapOutput:
    source_taps: tuple[torch.Tensor, ...]


class SourceTapProvider(nn.Module):
    """Execute the small-model trunk used by the matched cross-scale baseline."""

    def __init__(
        self,
        source: nn.Module,
        *,
        source_layers: int,
        tap_layers: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.source = source
        self.source_layers = source_layers
        self.tap_layers = tap_layers
        if tuple(sorted(set(tap_layers))) != tap_layers:
            raise ValueError("source taps must be sorted and unique")
        if source_layers <= 0 or source_layers > len(source.model.layers):
            raise ValueError("source layer count lies outside the model")
        if not tap_layers or tap_layers[0] < 0 or tap_layers[-1] >= source_layers:
            raise ValueError("source tap lies outside retained layers")

    def forward(
        self,
        input_ids: torch.LongTensor,
        *,
        cache_state: DualCacheState,
    ) -> SourceTapOutput:
        if not cache_state.cycle_active:
            raise RuntimeError("source forward requires an active cache transaction")
        if cache_state.proposed_tokens != input_ids.shape[1]:
            raise ValueError("input length must equal the registered proposal")
        offset = cache_state.committed_length
        positions = torch.arange(
            offset,
            offset + input_ids.shape[1],
            device=input_ids.device,
        )
        position_ids = positions.unsqueeze(0)
        hidden = self.source.model.embed_tokens(input_ids)
        mask = causal_mask(
            batch_size=input_ids.shape[0],
            query_length=input_ids.shape[1],
            position_offset=offset,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        rotary = self.source.model.rotary_emb(hidden, position_ids)
        taps: list[torch.Tensor] = []
        for index, layer in enumerate(self.source.model.layers[: self.source_layers]):
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_embeddings=rotary,
                position_ids=position_ids,
                past_key_values=cache_state.source_cache,
                use_cache=True,
                cache_position=positions,
            )
            if index in self.tap_layers:
                taps.append(hidden)
        return SourceTapOutput(source_taps=tuple(taps))
