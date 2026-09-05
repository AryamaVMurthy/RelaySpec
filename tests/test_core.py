from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def test_acceptance_stops_at_first_mismatch() -> None:
    from relayspec.generation import accepted_block_length

    block = torch.tensor([[4, 5, 6, 7]])
    posterior = torch.tensor([[5, 6, 7, 0]])
    assert accepted_block_length(block, posterior) == 4

    posterior[0, 1] = 8
    assert accepted_block_length(block, posterior) == 2


class FakeCache:
    def __init__(self) -> None:
        self.length = 0

    def crop(self, length: int) -> None:
        self.length = length


def test_dual_cache_commits_only_accepted_prefix(monkeypatch) -> None:
    from relayspec.cache import DualCacheState

    state = DualCacheState(FakeCache(), FakeCache())
    state.begin_cycle(16)
    state.commit(5)

    assert state.committed_length == 5
    assert state.source_cache.length == state.target_cache.length == 5
    assert not state.cycle_active


def test_source_tap_provider_validates_layers() -> None:
    from relayspec.source import SourceTapProvider

    source = SimpleNamespace(model=SimpleNamespace(layers=[object()] * 4))
    with pytest.raises(ValueError, match="source tap"):
        SourceTapProvider(source, source_layers=4, tap_layers=(1, 4))
