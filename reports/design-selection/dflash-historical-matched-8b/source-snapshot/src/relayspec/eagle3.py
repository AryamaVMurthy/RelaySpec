from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from relayspec.proposers import ContextProvider


@dataclass(frozen=True)
class OfficialEagle3API:
    model_class: type
    proposal_class: type
    generate_decoding_sample: Callable[..., Any]
    logits_to_probs: Callable[[torch.Tensor, float], torch.Tensor]
    sample_tokens: Callable[[torch.Tensor, float], torch.Tensor]


def import_official_eagle3(
    source_dir: str | Path,
    expected_commit: str,
) -> OfficialEagle3API:
    source = Path(source_dir).resolve()
    if not (source / ".git").is_dir():
        raise RuntimeError(f"DeepSpec source is not a Git checkout: {source}")
    actual_commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"DeepSpec source commit mismatch: expected {expected_commit}, "
            f"found {actual_commit}"
        )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    evaluator = importlib.import_module("deepspec.eval.base_evaluator")
    model = importlib.import_module("deepspec.modeling.eagle3.qwen3")
    sampling = importlib.import_module("deepspec.utils.sampling")
    return OfficialEagle3API(
        model_class=model.Qwen3Eagle3Model,
        proposal_class=evaluator.DraftProposal,
        generate_decoding_sample=evaluator.generate_decoding_sample,
        logits_to_probs=sampling.logits_to_probs,
        sample_tokens=sampling.sample_tokens,
    )


def _default_cache_factory() -> Any:
    from transformers import DynamicCache

    return DynamicCache()


def _default_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature < 1e-5:
        return logits.argmax(dim=-1)
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    return torch.multinomial(flat, 1).reshape(probabilities.shape[:-1])


def _default_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature < 1e-5:
        probabilities = torch.zeros_like(logits, dtype=torch.float32)
        probabilities.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return probabilities
    return torch.softmax(logits.float() / temperature, dim=-1)


def _default_proposal_factory(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


class ProfiledTarget:
    """Transparent target-model proxy that separates prefill from verification.

    DeepSpec calls the target once to prefill the prompt and once per speculative
    block to verify it.  Keeping this distinction at the model boundary avoids
    attributing Python bookkeeping or draft work to the full target.
    """

    def __init__(self, target_model: Any, profile_recorder: Any) -> None:
        self._target_model = target_model
        self._profile_recorder = profile_recorder
        self._calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target_model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        region = (
            "prefill_full_target" if self._calls == 0 else "verification_full_target"
        )
        self._calls += 1
        with self._profile_recorder.region(region):
            return self._target_model(*args, **kwargs)


class Eagle3Backend:
    """Pinned DeepSpec EAGLE-3 proposal state with a replaceable context provider."""

    def __init__(
        self,
        *,
        draft: torch.nn.Module,
        context_provider: ContextProvider,
        temperature: float = 0.0,
        max_proposal_tokens: int | None = None,
        cache_factory: Callable[[], Any] = _default_cache_factory,
        proposal_factory: Callable[..., Any] = _default_proposal_factory,
        sample_tokens: Callable[[torch.Tensor, float], torch.Tensor] = _default_sample,
        logits_to_probs: Callable[[torch.Tensor, float], torch.Tensor] = _default_probs,
        profile_recorder: Any | None = None,
    ) -> None:
        self.draft = draft
        self.context_provider = context_provider
        self.temperature = float(temperature)
        released_max = int(draft.ttt_length)
        selected = released_max if max_proposal_tokens is None else max_proposal_tokens
        if not 1 <= int(selected) <= released_max:
            raise ValueError("EAGLE proposal length must lie within the released cap")
        self.max_proposal_tokens = int(selected)
        self.cache_factory = cache_factory
        self.proposal_factory = proposal_factory
        self.sample_tokens = sample_tokens
        self.logits_to_probs = logits_to_probs
        self.profile_recorder = profile_recorder

    def _draft_region(self) -> Any:
        if self.profile_recorder is None:
            return nullcontext()
        return self.profile_recorder.region("draft")

    def initialize(
        self,
        *,
        initial_output: Any,
        output_ids: torch.Tensor,
        position_ids: torch.Tensor,
        num_input_tokens: int,
    ) -> SimpleNamespace:
        context_feature = self.context_provider.initialize(
            input_ids=output_ids[:, :num_input_tokens],
            target_output=initial_output,
        )
        expected_width = int(self.draft.fc.out_features)
        if context_feature.shape[-1] != expected_width:
            raise ValueError(
                "context provider must return the post-fusion EAGLE interface "
                f"width {expected_width}, found {context_feature.shape[-1]}"
            )
        shifted_prompt_ids = torch.cat(
            [
                output_ids[:, 1:num_input_tokens],
                output_ids[:, num_input_tokens : num_input_tokens + 1],
            ],
            dim=1,
        )
        draft_cache = self.cache_factory()
        with self._draft_region():
            draft_hidden = self.draft.extend_draft_cache(
                hidden_states=context_feature,
                input_ids=shifted_prompt_ids,
                position_ids=position_ids[:, :num_input_tokens],
                past_key_values=draft_cache,
            )
        return SimpleNamespace(
            draft_cache=draft_cache,
            draft_hidden=draft_hidden,
            position_ids=position_ids,
            current_pos=num_input_tokens,
            cache_len_before=0,
            last_verify_input_ids=None,
        )

    def propose(
        self,
        *,
        context: SimpleNamespace,
        output_ids: torch.Tensor,
        position_ids: torch.Tensor,
        start: int,
        stop_token_ids: list[int] | None = None,
    ) -> Any:
        del position_ids
        context.cache_len_before = context.draft_cache.get_seq_length()
        candidate_ids = [output_ids[:, start : start + 1]]
        draft_logits_list: list[torch.Tensor] = []
        proposal_hidden = context.draft_hidden
        next_position = start
        with self._draft_region():
            for _ in range(self.max_proposal_tokens):
                draft_logits = self.draft.compute_logits(proposal_hidden)
                draft_logits_list.append(draft_logits)
                next_token = self.sample_tokens(draft_logits, self.temperature)
                candidate_ids.append(next_token[:, -1:])
                if stop_token_ids is not None:
                    stops = torch.tensor(stop_token_ids, device=next_token.device)
                    if torch.isin(next_token, stops).any().item():
                        break
                proposal_hidden = self.draft(
                    hidden_states=proposal_hidden,
                    input_ids=next_token[:, -1:],
                    position_ids=context.position_ids[
                        :, next_position : next_position + 1
                    ],
                    past_key_values=context.draft_cache,
                    use_cache=True,
                )
                next_position += 1
            draft_logits = torch.cat(draft_logits_list, dim=1)
        verify_input_ids = torch.cat(candidate_ids, dim=1)
        context.last_verify_input_ids = verify_input_ids
        return self.proposal_factory(
            draft_token_count=int(draft_logits.shape[1]),
            verify_input_ids=verify_input_ids,
            draft_probs=self.logits_to_probs(draft_logits, self.temperature),
        )

    def update(self, context: SimpleNamespace, verification: Any) -> None:
        if verification.committed_tokens is None:
            raise ValueError("verification must expose committed_tokens")
        if context.last_verify_input_ids is None:
            raise RuntimeError("update requires a preceding proposal")
        committed_length = int(verification.committed_tokens.shape[1])
        context.draft_cache.crop(int(context.cache_len_before))
        context_feature = self.context_provider.update(
            verification_input_ids=context.last_verify_input_ids,
            target_output=verification.target_output,
            committed_length=committed_length,
        )
        with self._draft_region():
            context.draft_hidden = self.draft.extend_draft_cache(
                hidden_states=context_feature,
                input_ids=verification.committed_tokens,
                position_ids=context.position_ids[
                    :, context.current_pos : context.current_pos + committed_length
                ],
                past_key_values=context.draft_cache,
            )
        context.current_pos += committed_length


def eagle3_generate(
    *,
    api: OfficialEagle3API,
    target_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    context_provider: ContextProvider,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    stop_token_ids: list[int] | None,
    max_proposal_tokens: int | None = None,
    profile_recorder: Any | None = None,
) -> Any:
    backend = Eagle3Backend(
        draft=draft_model,
        context_provider=context_provider,
        temperature=temperature,
        max_proposal_tokens=max_proposal_tokens,
        proposal_factory=api.proposal_class,
        sample_tokens=api.sample_tokens,
        logits_to_probs=api.logits_to_probs,
        profile_recorder=profile_recorder,
    )
    profiled_target = (
        ProfiledTarget(target_model, profile_recorder)
        if profile_recorder is not None
        else target_model
    )
    result = api.generate_decoding_sample(
        target_model=profiled_target,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        max_proposal_tokens=backend.max_proposal_tokens,
        temperature=temperature,
        stop_token_ids=stop_token_ids,
        init_context=backend.initialize,
        propose=backend.propose,
        update=backend.update,
    )
    if hasattr(context_provider, "executed_tokens"):
        result.source_executed_tokens = int(context_provider.executed_tokens)
    return result
