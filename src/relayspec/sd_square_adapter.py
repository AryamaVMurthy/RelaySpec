"""Pinned SD-square loading and observation of its unmodified acceptance decisions."""

import ast
import hashlib
import sys
import types
from pathlib import Path


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
