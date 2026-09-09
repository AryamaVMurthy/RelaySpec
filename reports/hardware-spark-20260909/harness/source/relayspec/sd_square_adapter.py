"""Pinned SD-square loading and observation of its unmodified acceptance decisions."""

import ast
import hashlib
import sys
import types
from pathlib import Path


def steering_digest(named):
    """Fingerprint names, shapes, storage dtypes and tensor bytes in declared order."""
    import torch

    result = hashlib.sha256()
    for name, value in named:
        result.update(name.encode())
        result.update(str((tuple(value.shape), value.dtype)).encode())
        result.update(
            value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return result.hexdigest()


def load_inference_steering(named, saved, training_sha=None):
    """Verify training bytes before casting, then verify actual inference bytes."""
    import torch

    if set(saved) != {n for n, _ in named}:
        raise ValueError("SD-square campaign steering names changed")
    original_sha = steering_digest([(n, saved[n]) for n, _ in named])
    if training_sha is not None and original_sha != training_sha:
        raise ValueError("SD-square training checkpoint fingerprint changed")
    if any(p.dtype != torch.bfloat16 for _, p in named):
        raise ValueError("SD-square inference steering must use BF16 storage")
    converted = [(n, saved[n].to(dtype=torch.bfloat16)) for n, _ in named]
    expected_sha = steering_digest(converted)
    with torch.no_grad():
        for (_, parameter), (name, value) in zip(named, converted, strict=True):
            if parameter.shape != value.shape:
                raise ValueError(f"SD-square steering shape changed: {name}")
            parameter.copy_(value)
    actual_sha = steering_digest(named)
    if actual_sha != expected_sha:
        raise ValueError("SD-square inference weights differ from BF16 conversion")
    return {"training_sha256": training_sha, "inference_sha256": actual_sha}


def load_sd_square(source, config, model_paths):
    source = Path(source).resolve()
    actual = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(source.rglob("*.py"))
    }
    if actual != config["source_sha256"]:
        raise ValueError("unreviewed SD-square source")
    tree = ast.parse((source / "main.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "TrainingModule"
    )
    step = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_spec_dec_step"
    )
    positions = [
        i
        for i, n in enumerate(step.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "has_ended" for t in n.targets)
    ]
    if len(positions) != 1:
        raise ValueError("SD-square acceptance observer insertion point changed")
    step.body.insert(
        positions[0],
        ast.parse(
            "_relayspec_capture(self, input_ids, curr_pos, NA, next_token, v_logits, has_ended, attention_mask, position_ids)"
        ).body[0],
    )
    ast.fix_missing_locations(tree)
    module = types.ModuleType("relayspec_sd_square_upstream")
    module.__file__ = str(source / "main.py")
    sys.modules[module.__name__] = module
    sys.path.insert(0, str(source))
    exec(compile(tree, module.__file__, "exec"), module.__dict__)
    module.PRETTY_PRINT = False

    module._relayspec_capture = capture_sd_square_step
    # Only model/tokenizer resolution changes. Public architectures, dtypes,
    # training loss, guidance, cache layout and rejection sampling are retained.
    import torch
    from src.models.qwen import Qwen3ForCausalLM
    from transformers import AutoTokenizer

    def pinned_model(name, float16=False):
        keys = [
            k for k in ("target", "drafter") if name.lower() == config[k]["id"].lower()
        ]
        if len(keys) != 1:
            raise ValueError("undeclared SD-square model request")
        return Qwen3ForCausalLM.from_pretrained(
            model_paths[keys[0]],
            torch_dtype=torch.float16 if float16 else torch.float32,
            local_files_only=True,
        ), "qwen"

    class PinnedTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            if name.lower() != config["target"]["id"].lower():
                raise ValueError("undeclared SD-square tokenizer request")
            return AutoTokenizer.from_pretrained(
                model_paths["target"], local_files_only=True, **kwargs
            )

    module.load_model = pinned_model
    module.AutoTokenizer = PinnedTokenizer
    return module


def capture_sd_square_step(
    model, ids, curr, accepted, next_token, logits, ended, mask, positions
):
    if ids.shape[0] != 1:
        raise ValueError("acceptance observer currently requires batch one")
    if getattr(model, "_relayspec_capture_tensors", False):
        if not model.greedy_sample:
            raise ValueError(
                "deferred observation currently verifies greedy decoding only"
            )
        model._relayspec_tensor_trace.append(
            {
                "ended": ended.detach().clone(),
                "accepted": accepted.detach().clone(),
                "proposed": ids[0, curr + 1 : curr + model.NG + 1].detach().clone(),
                "next_token": next_token.detach().clone(),
                "argmax": logits[0].argmax(-1).detach(),
            }
        )
        return
    if bool(ended[0]):
        return
    count = int(accepted[0])
    proposed = ids[0, curr + 1 : curr + count + 1].tolist()
    committed = proposed + [int(next_token[0, 0])]
    target = logits[0, : count + 1].argmax(-1).tolist()
    if model.greedy_sample and committed != target:
        raise ValueError(
            "SD-square committed a token differing from its greedy verifier"
        )
    model._relayspec_trace.append(
        {
            "accepted_draft_tokens": count,
            "tokens": committed,
            "verifier_argmax": target,
        }
    )
    if getattr(model, "_relayspec_detailed_trace", False):
        values, indices = logits[0].float().topk(5, dim=-1)
        model._relayspec_trace[-1].update(
            logits_float32_sha256=hashlib.sha256(
                logits[0].float().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
            output_start=sum(len(r["tokens"]) for r in model._relayspec_trace[:-1]),
            top_ids=indices.cpu().tolist(),
            top_scores=values.cpu().tolist(),
            physical_cache_prefix=curr,
            valid_cached_prefix_ids=ids[0, :curr][mask[0, :curr].bool()].cpu().tolist(),
            query_ids=ids[0, curr : curr + model.NG + 1].cpu().tolist(),
            query_positions=positions[0, curr : curr + model.NG + 1].cpu().tolist(),
        )


def committed_tokens(trace, cap, eos):
    raw = [token for block in trace for token in block["tokens"]]
    end = min(len(raw), cap)
    if eos in raw[:end]:
        end = raw.index(eos) + 1
    return raw[:end], raw


def termination_status(trace, cap, eos, ngram):
    """Explain the pinned public generate loop's fourfold physical-slot guard."""
    tokens, raw = committed_tokens(trace, cap, eos)
    if cap < 1 or ngram < 1 or not tokens:
        raise ValueError("invalid or empty SD-square generation")
    cycle_limit = (cap * 4 - 1) // (ngram + 1)
    if len(tokens) == cap or tokens[-1] == eos:
        reason = "cap" if len(tokens) == cap else "eos"
    elif len(trace) == cycle_limit:
        reason = "public_physical_slot_guard"
    else:
        reason = "unexplained_early_stop"
    return {
        "reason": reason,
        "counted_tokens": len(tokens),
        "raw_tokens": len(raw),
        "cycles": len(trace),
        "public_cycle_limit": cycle_limit,
        "requested_cap": cap,
    }


def finalize_sd_square_trace(model):
    """Materialize and verify captured GPU decisions after the request timer stops."""
    result = []
    for block in model._relayspec_tensor_trace:
        if bool(block["ended"][0]):
            continue
        count = int(block["accepted"][0])
        tokens = block["proposed"][:count].cpu().tolist() + [
            int(block["next_token"][0, 0])
        ]
        target = block["argmax"][: count + 1].cpu().tolist()
        if tokens != target:
            raise ValueError(
                "SD-square committed a token differing from its greedy verifier"
            )
        result.append(
            {
                "accepted_draft_tokens": count,
                "tokens": tokens,
                "verifier_argmax": target,
            }
        )
    model._relayspec_trace = result
    return result


def collate_sd_square_records(records, pad_token_id):
    """Right-pad cached token records and retain the public default token loss mask."""
    import torch

    if not records or any(
        r.ndim != 2 or r.shape[0] != 1 or not 1 < r.shape[1] <= 192 for r in records
    ):
        raise ValueError(
            "SD-square collation requires nonempty single-sequence records capped at192"
        )
    if any(r.dtype != torch.long or r.device != records[0].device for r in records):
        raise ValueError("SD-square records differ in dtype or device")
    lengths = [r.shape[1] for r in records]
    targets = records[0].new_full((len(records), max(lengths)), pad_token_id)
    mask = torch.zeros_like(targets, dtype=torch.float32)
    for i, record in enumerate(records):
        targets[i, : lengths[i]] = record[0]
        mask[i, : lengths[i]] = record[0].ne(pad_token_id).float()
        mask[i, 0] = 1.0  # Same first-position rule as public process_batch.
    return {"targets": targets, "loss_mask": mask}, lengths
