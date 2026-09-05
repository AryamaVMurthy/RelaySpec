"""Instrument the pinned public PARD loop without replacing its proposal logic."""

import ast
import hashlib
import types

PARD_COMMIT = "6f279bf3f1680e0b5d71c562ca5b91bdeef4c038"
PARD_SOURCE_SHA256 = "9c4ae104e90ccb6a0a3948d011ab5e892cb7733894f812b74ab63270907c5340"


def instrument_source(source, *, trace_targets=False):
    """Add request timing and raw-token capture to the exact reviewed source.

    All upstream statements remain in their original order. Timings include
    both model prefills and cache reset, and end before output detokenization.
    The upstream aggregate TPS excludes its first block's time and is unused.
    """
    if hashlib.sha256(source).hexdigest() != PARD_SOURCE_SHA256:
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
    if trace_targets:
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
        while_loop.body[index:index] = ast.parse(
            "self._record_target_trace(all_token, input_token_lenght, target_input, "
            "target_output.logits[:, -draft_tmp_new_token.shape[1] - 1:])\n"
        ).body
    return ast.fix_missing_locations(tree)


def load_instrumented_pard(source_path, *, trace_targets=False):
    source = source_path.read_bytes()
    tree = instrument_source(source, trace_targets=trace_targets)
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

        def _record_request(self, all_token, input_length, seconds, profile):
            if self.captured_request is not None:
                raise ValueError(
                    "capture interface expects exactly one request per call"
                )
            self.captured_request = {
                "token_ids": all_token[:, input_length:].detach(),
                "request_seconds": seconds,
                "accepted_lengths": list(profile["accept_length"]),
                "draft_calls": len(profile["draft"]),
                "target_calls": len(profile["accept_length"]),
                "target_trace": self._target_trace,
            }

    return CapturedPard


def trim_pard_tokens(token_ids, *, max_new_tokens, eos_token_id):
    """Count only the requested prefix; retain raw overshoot for the audit."""
    if max_new_tokens <= 0:
        raise ValueError("positive output cap required")
    values = list(token_ids[:max_new_tokens])
    if eos_token_id in values:
        values = values[: values.index(eos_token_id) + 1]
    return values
