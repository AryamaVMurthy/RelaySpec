"""Run the existing benchmark with target-call traces, outside timing evidence."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import yaml
from transformers import AutoModelForCausalLM


def main() -> None:
    config_path = Path(sys.argv[sys.argv.index("--config") + 1])
    config = yaml.safe_load(config_path.read_text())
    family = config["proposer"]["family"]
    script = "benchmark_relay.py" if family == "dflash" else "benchmark_eagle3.py"
    spec = importlib.util.spec_from_file_location(
        "traced_benchmark", Path(__file__).with_name(script)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    pending = {}

    def before(model, args, kwargs):
        ids = args[0] if args else kwargs["input_ids"]
        cache = kwargs.get("past_key_values")
        pending.clear()
        pending.update(
            {
                "input_ids": ids.detach().cpu().tolist(),
                "cache_before": int(cache.get_seq_length()) if cache is not None else 0,
                "position_ids": kwargs["position_ids"].detach().cpu().tolist()
                if kwargs.get("position_ids") is not None
                else None,
                "attention_mask_shape": list(kwargs["attention_mask"].shape)
                if kwargs.get("attention_mask") is not None
                else None,
                "output_hidden_states": bool(kwargs.get("output_hidden_states", False)),
                "logits_to_keep": kwargs.get("logits_to_keep"),
            }
        )

    def after(model, args, kwargs, output):
        top = output.logits.float().topk(2, dim=-1)
        calls.append(
            {
                **pending,
                "top_ids": top.indices.cpu().tolist(),
                "top_logits": top.values.cpu().tolist(),
                "chosen_ids": output.logits.argmax(-1).cpu().tolist(),
                "cache_after": int(output.past_key_values.get_seq_length()),
            }
        )

    original_load = AutoModelForCausalLM.from_pretrained

    def load(model_id, *args, **kwargs):
        model = original_load(model_id, *args, **kwargs)
        if model_id == config["target"]["id"]:
            model.register_forward_pre_hook(before, with_kwargs=True)
            model.register_forward_hook(after, with_kwargs=True)
        return model

    AutoModelForCausalLM.from_pretrained = load
    original_run = module.run_method
    output_dir = Path(os.environ["RELAYSPEC_OUTPUT"])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"trace-rank{os.environ['RANK']}.jsonl"
    with path.open("x") as stream:

        def run(name, generate, input_ids, max_new_tokens, tokenizer):
            calls.clear()
            result = {}

            def capture(**kwargs):
                stats = generate(**kwargs)
                result["output_ids"] = stats.output_ids.detach().cpu().tolist()
                return stats

            row = original_run(name, capture, input_ids, max_new_tokens, tokenizer)
            stream.write(
                json.dumps(
                    {
                        "method": name,
                        "input_ids": input_ids.cpu().tolist(),
                        **result,
                        "calls": calls,
                        "timing_scope": "instrumented diagnostic, not benchmark evidence",
                    }
                )
                + "\n"
            )
            stream.flush()
            return row

        module.run_method = run
        module.main()


if __name__ == "__main__":
    main()
