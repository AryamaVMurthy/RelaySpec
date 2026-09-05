"""Small CPU checks for the submission review; never loads remote model weights.

From the repository root:
  .venv/bin/python reports/submission-review-2026-09-05/reproduce_checks.py

Add --checkpoint to exercise the current drift export helper. This needs PEFT;
the review used an isolated dependency directory rather than changing .venv:
  PYTHONPATH=tmp/submission-review-2026-09-05/deps .venv/bin/python \
    reports/submission-review-2026-09-05/reproduce_checks.py --checkpoint

Printed numbers may differ from the original archived random toy example.
The diagnostic conclusions, not its particular random scores, are the checks.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import tempfile
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]


def objective_check() -> dict[str, float]:
    torch.manual_seed(1729)
    features = torch.randn(80, 7, dtype=torch.float64)
    teacher = torch.randn(80, 5, dtype=torch.float64)
    teacher = torch.nn.functional.rms_norm(teacher, (5,))
    energy = teacher.square().sum(-1)
    row_scale = energy.rsqrt().unsqueeze(-1)
    weight = torch.linalg.lstsq(features * row_scale, teacher * row_scale).solution
    weight = weight.detach().requires_grad_()
    prediction = features @ weight
    linear_loss = ((prediction - teacher).square().sum(-1) / energy).mean()
    normalized_prediction = torch.nn.functional.rms_norm(prediction, (5,))
    actual_loss = ((normalized_prediction - teacher).square().sum(-1) / energy).mean()
    linear_gradient = torch.autograd.grad(linear_loss, weight, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual_loss, weight)[0]
    return {
        "linear_objective_max_abs_gradient": linear_gradient.abs().max().item(),
        "post_normalization_objective_max_abs_gradient": actual_gradient.abs().max().item(),
    }


def checkpoint_check() -> dict[str, Any]:
    import peft
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

    # Extract only the repository's current helper to avoid importing its data
    # acquisition and distributed-training dependencies. No code is rewritten.
    source = ROOT / "scripts/lora_sft_drift.py"
    module = ast.parse(source.read_text())
    helper = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                  and node.name == "save_merged_checkpoint")
    scope: dict[str, Any] = {"Path": Path, "Any": Any}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), scope)

    class TokenizerStub:
        def save_pretrained(self, path: Path) -> None:
            pass  # The model-weight round trip is independent of tokenizer files.

    torch.manual_seed(1729)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8,
                        max_position_embeddings=64, attention_dropout=0.0)
    model = get_peft_model(Qwen3ForCausalLM(config), LoraConfig(
        r=2, lora_alpha=4, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0, task_type="CAUSAL_LM",
    )).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        before = model(ids).logits

    result: dict[str, Any] = {"peft": peft.__version__, "transformers": transformers.__version__,
                              "torch": torch.__version__}
    with tempfile.TemporaryDirectory(prefix="relayspec-review-") as directory:
        bad_path = Path(directory) / "current-export"
        scope["save_merged_checkpoint"](model, TokenizerStub(), bad_path)
        loaded, info = AutoModelForCausalLM.from_pretrained(bad_path, output_loading_info=True)
        loaded.eval()
        with torch.no_grad():
            difference = (before - loaded(ids).logits).abs().max().item()
        result["current_export"] = {
            "missing": sorted(info["missing_keys"]), "unexpected": sorted(info["unexpected_keys"]),
            "max_abs_logit_difference": difference,
        }
        good_path = Path(directory) / "unloaded-export"
        copy.deepcopy(model).merge_and_unload().save_pretrained(good_path)
        loaded, info = AutoModelForCausalLM.from_pretrained(good_path, output_loading_info=True)
        loaded.eval()
        with torch.no_grad():
            difference = (before - loaded(ids).logits).abs().max().item()
        result["unloaded_export"] = {
            "missing": sorted(info["missing_keys"]), "unexpected": sorted(info["unexpected_keys"]),
            "max_abs_logit_difference": difference,
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()
    result = {"objective_counterexample": objective_check(), "parameter_counts": {
        "adapter_14b_into_8b": 25600 * 20480,
        "direct_14b": 25600 * 2560,
        "linear_8b": 20480 * 2560,
        "nonlinear_512": 20480 * 512 + 512 * 2560,
    }}
    if args.checkpoint:
        result["checkpoint"] = checkpoint_check()
    print(json.dumps(result, indent=2))
