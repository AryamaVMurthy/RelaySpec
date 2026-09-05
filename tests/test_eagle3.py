from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
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


class FakeDraft(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ttt_length = 2
        self.target_layer_ids = [1, 9, 17, 25, 33]
        self.fc = torch.nn.Linear(10, 2, bias=False)
        self.extend_calls: list[dict[str, torch.Tensor]] = []

    def extend_draft_cache(
        self, *, hidden_states, input_ids, position_ids, past_key_values
    ):
        self.extend_calls.append(
            {
                "hidden_states": hidden_states.clone(),
                "input_ids": input_ids.clone(),
                "position_ids": position_ids.clone(),
            }
        )
        past_key_values.length += input_ids.shape[1]
        return hidden_states[:, -1:, :]

    def compute_logits(self, hidden_states):
        logits = torch.zeros((*hidden_states.shape[:2], 5))
        logits[..., 3] = 1
        return logits

    def forward(
        self, *, hidden_states, input_ids, position_ids, past_key_values, use_cache
    ):
        assert use_cache
        past_key_values.length += input_ids.shape[1]
        return hidden_states + 1


class RecordingProvider:
    def __init__(self, initial_context: torch.Tensor) -> None:
        self.initial_context = initial_context
        self.update_calls: list[tuple[torch.Tensor, int]] = []

    def initialize(self, *, input_ids, target_output):
        return self.initial_context

    def update(
        self,
        *,
        verification_input_ids,
        target_output,
        committed_length,
    ):
        self.update_calls.append((verification_input_ids.clone(), committed_length))
        return self.initial_context[:, :committed_length] + 2


class RecordingProfiler:
    def __init__(self) -> None:
        self.names: list[str] = []

    @contextmanager
    def region(self, name: str):
        self.names.append(name)
        yield


def test_eagle3_initialize_uses_postfusion_context_and_shifted_tokens() -> None:
    from relayspec.eagle3 import Eagle3Backend

    draft = FakeDraft()
    provider = RecordingProvider(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]))
    backend = Eagle3Backend(
        draft=draft,
        context_provider=provider,
        cache_factory=FakeCache,
    )
    output_ids = torch.tensor([[10, 11, 12]])
    positions = torch.arange(8).unsqueeze(0)

    context = backend.initialize(
        initial_output=SimpleNamespace(hidden_states=()),
        output_ids=output_ids,
        position_ids=positions,
        num_input_tokens=2,
    )

    call = draft.extend_calls[0]
    assert call["hidden_states"].shape[-1] == 2
    assert call["input_ids"].tolist() == [[11, 12]]
    assert call["position_ids"].tolist() == [[0, 1]]
    assert context.current_pos == 2


def test_eagle3_update_crops_speculation_and_uses_committed_prefix() -> None:
    from relayspec.eagle3 import Eagle3Backend

    draft = FakeDraft()
    provider = RecordingProvider(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]))
    backend = Eagle3Backend(
        draft=draft,
        context_provider=provider,
        cache_factory=FakeCache,
    )
    output_ids = torch.tensor([[10, 11, 12]])
    positions = torch.arange(8).unsqueeze(0)
    context = backend.initialize(
        initial_output=SimpleNamespace(hidden_states=()),
        output_ids=output_ids,
        position_ids=positions,
        num_input_tokens=2,
    )
    context.cache_len_before = 2
    context.draft_cache.length = 7
    context.last_verify_input_ids = torch.tensor([[12, 13, 14, 15]])
    verification = SimpleNamespace(
        target_output=SimpleNamespace(hidden_states=()),
        committed_tokens=torch.tensor([[13, 99]]),
    )

    backend.update(context, verification)

    assert provider.update_calls[0][0].tolist() == [[12, 13, 14, 15]]
    assert provider.update_calls[0][1] == 2
    assert draft.extend_calls[-1]["input_ids"].tolist() == [[13, 99]]
    assert context.current_pos == 4


def test_import_official_eagle3_rejects_wrong_commit(tmp_path: Path) -> None:
    from relayspec.eagle3 import import_official_eagle3

    with pytest.raises(RuntimeError, match="Git checkout"):
        import_official_eagle3(tmp_path, "expected")


def test_profiled_target_labels_prefill_then_verification() -> None:
    from relayspec.eagle3 import ProfiledTarget

    class Target:
        config = SimpleNamespace(name="target")

        def __call__(self, value):
            return value + 1

    profiler = RecordingProfiler()
    target = ProfiledTarget(Target(), profiler)

    assert target(1) == 2
    assert target(2) == 3
    assert target.config.name == "target"
    assert profiler.names == ["prefill_full_target", "verification_full_target"]


def test_eagle3_backend_profiles_only_draft_operations() -> None:
    from relayspec.eagle3 import Eagle3Backend

    profiler = RecordingProfiler()
    draft = FakeDraft()
    provider = RecordingProvider(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]))
    backend = Eagle3Backend(
        draft=draft,
        context_provider=provider,
        cache_factory=FakeCache,
        profile_recorder=profiler,
    )
    output_ids = torch.tensor([[10, 11, 12]])
    positions = torch.arange(8).unsqueeze(0)
    context = backend.initialize(
        initial_output=SimpleNamespace(hidden_states=()),
        output_ids=output_ids,
        position_ids=positions,
        num_input_tokens=2,
    )
    backend.propose(
        context=context,
        output_ids=output_ids,
        position_ids=positions,
        start=2,
    )

    assert profiler.names == ["draft", "draft"]


def test_eagle3_generate_profiles_target_and_exports_source_work() -> None:
    from relayspec.eagle3 import OfficialEagle3API, eagle3_generate

    class Provider(RecordingProvider):
        executed_tokens = 17

    class Target:
        def __call__(self, value):
            return value + 1

    def generate_decoding_sample(**kwargs):
        assert kwargs["target_model"](1) == 2
        assert kwargs["target_model"](2) == 3
        return SimpleNamespace(output_ids=torch.tensor([[1, 2]]))

    api = OfficialEagle3API(
        model_class=object,
        proposal_class=lambda **kwargs: SimpleNamespace(**kwargs),
        generate_decoding_sample=generate_decoding_sample,
        logits_to_probs=lambda logits, temperature: logits,
        sample_tokens=lambda logits, temperature: logits.argmax(dim=-1),
    )
    profiler = RecordingProfiler()
    result = eagle3_generate(
        api=api,
        target_model=Target(),
        draft_model=FakeDraft(),
        context_provider=Provider(torch.zeros((1, 1, 2))),
        input_ids=torch.tensor([[1]]),
        max_new_tokens=2,
        temperature=0.0,
        stop_token_ids=None,
        profile_recorder=profiler,
    )

    assert result.source_executed_tokens == 17
    assert profiler.names == ["prefill_full_target", "verification_full_target"]


def test_eagle3_benchmark_executes_every_conversation_turn() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "benchmark_eagle3.py"
    ).read_text(encoding="utf-8")

    assert "for turn_index, user_content in enumerate(turns):" in script
    assert "histories[method]" in script
    assert 'problem_id = f"{problem_id}/turn{turn_index}"' in script
    assert "prompt = benchmark_turns(record)[0]" not in script
