"""Instrument the pinned public PARD loop without replacing its proposal logic."""

import ast
import hashlib
import types

PARD_COMMIT = "6f279bf3f1680e0b5d71c562ca5b91bdeef4c038"
PARD_SOURCE_SHA256 = "9c4ae104e90ccb6a0a3948d011ab5e892cb7733894f812b74ab63270907c5340"

PARD2_SOURCE_SHA256 = "6172caec7133b554fc4f2d3c621f1467bd63d349d3f825d716adc8f19d77fca7"


def instrument_source(source, *, trace_targets=False, trace_decisions=False, version=1):
    """Add request timing and raw-token capture to the exact reviewed source.

    All upstream statements remain in their original order. Timings include
    both model prefills and cache reset, and end before output detokenization.
    The upstream aggregate TPS excludes its first block's time and is unused.
    """
    if version not in (1, 2) or hashlib.sha256(source).hexdigest() != (
        PARD_SOURCE_SHA256 if version == 1 else PARD2_SOURCE_SHA256
    ):
        raise ValueError("PARD source differs from the reviewed pinned implementation")
    tree = ast.parse(source)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PardInfer"
    )
    generate = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "generate"
    )
    loops = [
        n
        for n in ast.walk(generate)
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id == "text"
    ]
    if len(loops) != 1:
        raise ValueError("PARD request loop is ambiguous")
    loop = loops[0]
    capture_index = [
        i
        for i, n in enumerate(loop.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "output" for t in n.targets)
    ]
    if len(capture_index) != 1:
        raise ValueError("PARD detokenization boundary is ambiguous")
    capture = ast.parse(
        "torch.cuda.synchronize()\n"
        "_relayspec_request_seconds = time.perf_counter() - _relayspec_request_started\n"
        "self._record_request(all_token, input_token_lenght, _relayspec_request_seconds, profile)\n"
    ).body
    loop.body[capture_index[0] : capture_index[0]] = capture
    loop.body[:0] = ast.parse(
        "torch.cuda.synchronize()\n_relayspec_request_started = time.perf_counter()\n"
    ).body
    if trace_targets or trace_decisions:
        if trace_targets and trace_decisions:
            raise ValueError("choose detailed or deferred PARD observation")
        while_loop = next(n for n in loop.body if isinstance(n, ast.While))
        boundaries = [
            i
            for i, n in enumerate(while_loop.body)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "keep_token_ids" for t in n.targets
            )
            and isinstance(n.value, ast.List)
        ]
        if len(boundaries) != 1:
            raise ValueError("PARD verification boundary is ambiguous")
        index = boundaries[0]
        observation = (
            "self._record_target_trace(all_token, input_token_lenght, target_input, "
            "target_output.logits[:, -draft_tmp_new_token.shape[1] - 1:])\n"
            if trace_targets
            else "self._record_target_decisions(all_token, input_token_lenght, new_token_ids)\n"
        )
        while_loop.body[index:index] = ast.parse(observation).body
    return ast.fix_missing_locations(tree)


def load_instrumented_pard(
    source_path, *, trace_targets=False, trace_decisions=False, version=1
):
    source = source_path.read_bytes()
    tree = instrument_source(
        source,
        trace_targets=trace_targets,
        trace_decisions=trace_decisions,
        version=version,
    )
    module = types.ModuleType("relayspec_external_pard")
    module.__file__ = str(source_path)
    exec(compile(tree, str(source_path), "exec"), module.__dict__)
    module.logger.disable(module.__name__)

    class CapturedPard(module.PardInfer):
        """Use the benchmark's already-tokenized prompt and expose raw outputs."""

        def set_input_ids(self, input_ids):
            if input_ids.ndim != 2 or input_ids.shape[0] != 1:
                raise ValueError("PARD benchmark requires a single tokenized request")
            if (
                input_ids.shape[1] + self.tokens + self.draft_k + 32
                >= self.max_cache_len
            ):
                raise ValueError("PARD static cache would truncate this request")
            self._input_ids = input_ids
            self.captured_request = None
            self._target_trace = []

        def get_input(self, prompt, tokenizer, prompt_type):
            return prompt, self._input_ids

        def _record_target_trace(self, all_token, input_length, target_input, logits):
            values, indices = logits.float().topk(5, dim=-1)
            self._target_trace.append(
                {
                    "output_start": all_token.shape[1] - input_length,
                    "prefix_ids": all_token[0].cpu().tolist(),
                    "incoming_ids": target_input["input_ids"][0].cpu().tolist(),
                    "cache_position": target_input["cache_position"].cpu().tolist(),
                    "argmax_ids": logits.argmax(-1)[0].cpu().tolist(),
                    "top_ids": indices[0].cpu().tolist(),
                    "top_scores": values[0].cpu().tolist(),
                }
            )

        def _record_target_decisions(self, all_token, input_length, argmax_ids):
            self._target_trace.append(
                {
                    "output_start": all_token.shape[1] - input_length,
                    "argmax_ids": argmax_ids[0].detach().clone(),
                }
            )

        def _record_request(self, all_token, input_length, seconds, profile):
            if self.captured_request is not None:
                raise ValueError(
                    "capture interface expects exactly one request per call"
                )
            if trace_decisions:
                for entry in self._target_trace:
                    entry["argmax_ids"] = entry["argmax_ids"].cpu().tolist()
            self.captured_request = {
                "token_ids": all_token[:, input_length:].detach(),
                "request_seconds": seconds,
                "accepted_lengths": list(profile["accept_length"]),
                "draft_calls": len(profile["draft"]),
                "target_calls": len(profile["accept_length"])
                + int(version == 2 and self.v2),
                "target_trace": self._target_trace,
            }

    return CapturedPard


def verify_pard_decisions(raw, accepted_lengths, trace, *, draft_k=12):
    """Check every accepted prefix and bonus against the actual greedy target."""
    offset = 0
    if not trace or len(trace) != len(accepted_lengths):
        raise ValueError("PARD decision trace lacks accepted blocks")
    for length, decision in zip(accepted_lengths, trace, strict=True):
        if (
            not 1 <= length <= draft_k + 1
            or decision["output_start"] != offset
            or len(decision["argmax_ids"]) != draft_k + 1
            or raw[offset : offset + length] != decision["argmax_ids"][:length]
        ):
            raise ValueError("PARD committed output differs from its actual verifier")
        offset += length
    if offset != len(raw):
        raise ValueError("PARD trace does not cover the full raw output")


def trim_pard_tokens(token_ids, *, max_new_tokens, eos_token_id):
    """Count only the requested prefix; retain raw overshoot for the audit."""
    if max_new_tokens <= 0:
        raise ValueError("positive output cap required")
    values = list(token_ids[:max_new_tokens])
    if eos_token_id in values:
        values = values[: values.index(eos_token_id) + 1]
    return values
