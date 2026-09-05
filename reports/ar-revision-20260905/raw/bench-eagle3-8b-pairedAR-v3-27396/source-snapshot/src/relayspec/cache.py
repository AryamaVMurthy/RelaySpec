from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DualCacheState:
    source_cache: Any
    target_cache: Any
    committed_length: int = 0
    _proposed_tokens: int | None = field(default=None, init=False, repr=False)

    @classmethod
    def create(cls, source_config: Any, target_config: Any) -> DualCacheState:
        from transformers import DynamicCache

        return cls(
            source_cache=DynamicCache(config=source_config),
            target_cache=DynamicCache(config=target_config),
        )

    @property
    def cycle_active(self) -> bool:
        return self._proposed_tokens is not None

    @property
    def proposed_tokens(self) -> int | None:
        return self._proposed_tokens

    def begin_cycle(self, proposed_tokens: int) -> None:
        if self.cycle_active:
            raise RuntimeError("a cache transaction is already active")
        if proposed_tokens <= 0:
            raise ValueError("proposed_tokens must be positive")
        self._proposed_tokens = proposed_tokens

    def commit(self, accepted_tokens: int) -> None:
        if self._proposed_tokens is None:
            raise RuntimeError("no active cache transaction")
        if not 0 <= accepted_tokens <= self._proposed_tokens:
            raise ValueError("accepted_tokens must be within the proposed block")
        new_length = self.committed_length + accepted_tokens
        self.source_cache.crop(new_length)
        self.target_cache.crop(new_length)
        self.committed_length = new_length
        self._proposed_tokens = None
