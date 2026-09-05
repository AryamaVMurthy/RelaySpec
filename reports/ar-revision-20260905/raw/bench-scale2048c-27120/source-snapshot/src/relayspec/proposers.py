from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from relayspec.relay import extract_hidden_taps


@runtime_checkable
class ContextProvider(Protocol):
    """Supply the exact tensor consumed by a frozen proposer."""

    def initialize(
        self, *, input_ids: torch.Tensor, target_output: Any
    ) -> torch.Tensor: ...


def project_source_interface(
    draft: Any,
    source_features: torch.Tensor,
    *,
    family: str,
) -> torch.Tensor:
    """Apply the immutable source proposer's released conditioning interface."""
    if family == "dflash":
        return draft.hidden_norm(draft.fc(source_features))
    if family == "eagle3":
        return draft.project_hidden_states(source_features)
    raise ValueError(f"unsupported proposer family: {family}")

    def update(
        self,
        *,
        verification_input_ids: torch.Tensor,
        target_output: Any,
        committed_length: int,
    ) -> torch.Tensor: ...


@dataclass
class SourceCacheState:
    source_cache: Any
    committed_length: int = 0
    _proposed_tokens: int | None = field(default=None, init=False, repr=False)

    @property
    def cycle_active(self) -> bool:
        return self._proposed_tokens is not None

    @property
    def proposed_tokens(self) -> int | None:
        return self._proposed_tokens

    def begin_cycle(self, proposed_tokens: int) -> None:
        if self.cycle_active:
            raise RuntimeError("a source cache transaction is already active")
        if proposed_tokens <= 0:
            raise ValueError("proposed_tokens must be positive")
        self._proposed_tokens = int(proposed_tokens)

    def commit(self, accepted_tokens: int) -> None:
        if self._proposed_tokens is None:
            raise RuntimeError("no active source cache transaction")
        if not 0 <= accepted_tokens <= self._proposed_tokens:
            raise ValueError("accepted_tokens must be within the proposed block")
        self.committed_length += int(accepted_tokens)
        self.source_cache.crop(self.committed_length)
        self._proposed_tokens = None


class SourceContextProvider:
    """Reconstruct proposer context with an optimized frozen source trunk.

    Verification happens before ``update``. Consequently only the committed
    prefix is executed by the source trunk; rejected speculative suffixes are
    never useful to the next proposal and are not charged to this baseline.
    """

    def __init__(
        self,
        *,
        tap_provider: Any,
        source_cache: Any,
        project: Callable[[torch.Tensor], torch.Tensor],
        profile_recorder: Any | None = None,
    ) -> None:
        self.tap_provider = tap_provider
        self.cache_state = SourceCacheState(source_cache)
        self.project = project
        self.profile_recorder = profile_recorder
        self.executed_tokens = 0

    def _execute(self, input_ids: torch.Tensor, *, region_name: str) -> torch.Tensor:
        region = (
            self.profile_recorder.region(region_name)
            if self.profile_recorder is not None
            else nullcontext()
        )
        with region:
            length = int(input_ids.shape[1])
            self.cache_state.begin_cycle(length)
            output = self.tap_provider(input_ids, cache_state=self.cache_state)
            concatenated = torch.cat(output.source_taps, dim=-1)
            self.cache_state.commit(length)
            self.executed_tokens += length
            return self.project(concatenated)

    def initialize(
        self, *, input_ids: torch.Tensor, target_output: Any
    ) -> torch.Tensor:
        del target_output
        return self._execute(input_ids, region_name="prefill_source_trunk")

    def update(
        self,
        *,
        verification_input_ids: torch.Tensor,
        target_output: Any,
        committed_length: int,
    ) -> torch.Tensor:
        del target_output
        if not 0 < committed_length <= verification_input_ids.shape[1]:
            raise ValueError("committed_length must lie within verification_input_ids")
        return self._execute(
            verification_input_ids[:, :committed_length],
            region_name="verification_source_trunk",
        )


class RelayContextProvider:
    """Predict proposer context from hidden states already produced by the target."""

    def __init__(
        self,
        *,
        relay: torch.nn.Module,
        target_layer_ids: tuple[int, ...],
        postprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
        profile_recorder: Any | None = None,
    ) -> None:
        self.relay = relay
        self.target_layer_ids = target_layer_ids
        self.postprocess = postprocess or (lambda tensor: tensor)
        self.profile_recorder = profile_recorder

    def _translate(
        self, target_output: Any, length: int | None, *, region_name: str
    ) -> torch.Tensor:
        region = (
            self.profile_recorder.region(region_name)
            if self.profile_recorder is not None
            else nullcontext()
        )
        with region:
            if getattr(target_output, "hidden_states", None) is None:
                raise ValueError("target output must include hidden_states")
            features = extract_hidden_taps(
                target_output.hidden_states,
                self.target_layer_ids,
            )
            if length is not None:
                features = features[:, :length]
            return self.postprocess(self.relay(features))

    def initialize(
        self, *, input_ids: torch.Tensor, target_output: Any
    ) -> torch.Tensor:
        return self._translate(
            target_output,
            int(input_ids.shape[1]),
            region_name="prefill_relay",
        )

    def update(
        self,
        *,
        verification_input_ids: torch.Tensor,
        target_output: Any,
        committed_length: int,
    ) -> torch.Tensor:
        if not 0 < committed_length <= verification_input_ids.shape[1]:
            raise ValueError("committed_length must lie within verification_input_ids")
        return self._translate(
            target_output,
            committed_length,
            region_name="relay",
        )
