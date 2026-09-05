from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


class FakeCache:
    def __init__(self) -> None:
        self.length = 0

    def crop(self, length: int) -> None:
        self.length = int(length)

    def get_seq_length(self) -> int:
        return self.length


class RecordingTapProvider:
    tap_layers = (0,)

    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def __call__(self, input_ids, *, cache_state):
        assert cache_state.cycle_active
        self.calls.append(input_ids.clone())
        feature = input_ids.to(torch.float32).unsqueeze(-1)
        return SimpleNamespace(source_taps=(feature,))


class RecordingProfiler:
    def __init__(self) -> None:
        self.names: list[str] = []

    @contextmanager
    def region(self, name: str):
        self.names.append(name)
        yield


def test_source_context_provider_executes_only_committed_prefix() -> None:
    from relayspec.proposers import SourceContextProvider

    taps = RecordingTapProvider()
    provider = SourceContextProvider(
        tap_provider=taps,
        source_cache=FakeCache(),
        project=lambda features: features + 10,
    )

    initial = provider.initialize(
        input_ids=torch.tensor([[1, 2, 3]]),
        target_output=None,
    )
    updated = provider.update(
        verification_input_ids=torch.tensor([[4, 5, 6, 7, 8, 9]]),
        target_output=None,
        committed_length=3,
    )

    assert initial.squeeze(-1).tolist() == [[11.0, 12.0, 13.0]]
    assert updated.squeeze(-1).tolist() == [[14.0, 15.0, 16.0]]
    assert [call.tolist() for call in taps.calls] == [
        [[1, 2, 3]],
        [[4, 5, 6]],
    ]
    assert provider.executed_tokens == 6
    assert provider.cache_state.committed_length == 6


def test_relay_context_provider_uses_target_hidden_prefix() -> None:
    from relayspec.proposers import RelayContextProvider

    relay = torch.nn.Identity()
    provider = RelayContextProvider(
        relay=relay,
        target_layer_ids=(0, 2),
        postprocess=lambda tensor: tensor + 1,
    )
    hidden_states = tuple(torch.full((1, 4, 2), float(index)) for index in range(5))
    target_output = SimpleNamespace(hidden_states=hidden_states)

    initial = provider.initialize(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        target_output=target_output,
    )
    updated = provider.update(
        verification_input_ids=torch.tensor([[8, 9, 10, 11]]),
        target_output=target_output,
        committed_length=2,
    )

    assert initial.shape == (1, 4, 4)
    assert initial[0, 0].tolist() == [2.0, 2.0, 4.0, 4.0]
    assert torch.equal(updated, initial[:, :2])


def test_source_context_provider_rejects_invalid_commit_length() -> None:
    from relayspec.proposers import SourceContextProvider

    provider = SourceContextProvider(
        tap_provider=RecordingTapProvider(),
        source_cache=FakeCache(),
        project=lambda features: features,
    )
    provider.initialize(input_ids=torch.tensor([[1]]), target_output=None)

    with pytest.raises(ValueError, match="committed_length"):
        provider.update(
            verification_input_ids=torch.tensor([[2, 3]]),
            target_output=None,
            committed_length=3,
        )


def test_context_providers_label_prefill_and_update_regions() -> None:
    from relayspec.proposers import SourceContextProvider

    profiler = RecordingProfiler()
    provider = SourceContextProvider(
        tap_provider=RecordingTapProvider(),
        source_cache=FakeCache(),
        project=lambda features: features,
        profile_recorder=profiler,
    )
    provider.initialize(input_ids=torch.tensor([[1, 2]]), target_output=None)
    provider.update(
        verification_input_ids=torch.tensor([[3, 4]]),
        target_output=None,
        committed_length=1,
    )

    assert profiler.names == ["prefill_source_trunk", "verification_source_trunk"]


def test_provider_contract_is_runtime_checkable() -> None:
    from relayspec.proposers import ContextProvider

    provider = SimpleNamespace(initialize=lambda **_: None, update=lambda **_: None)

    assert isinstance(provider, ContextProvider)


def test_project_source_interface_matches_each_released_family() -> None:
    from relayspec.proposers import project_source_interface

    features = torch.tensor([[[1.0, 2.0]]])

    class DFlash:
        fc = staticmethod(lambda value: value + 1)
        hidden_norm = staticmethod(lambda value: value * 2)

    class Eagle:
        project_hidden_states = staticmethod(lambda value: value - 1)

    assert torch.equal(
        project_source_interface(DFlash(), features, family="dflash"),
        (features + 1) * 2,
    )
    assert torch.equal(
        project_source_interface(Eagle(), features, family="eagle3"),
        features - 1,
    )
