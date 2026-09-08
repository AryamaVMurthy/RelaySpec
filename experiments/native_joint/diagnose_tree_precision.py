"""Replay first divergent predictions, then compare full FP32 native/tree decoding."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import cast_parameters, token_ids
from radical_decode import decode_variant, tree_layout
from run_lane import TARGET, DRAFT, COMMIT, sha


def first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


def prediction_step(progress, position):
    start = 0
    for step, count in enumerate(progress):
        if start < position <= start+count:
            return step, start, position-start-1
        start += count
    raise ValueError("Prediction is outside recorded committed progress")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg["variant"]["kind"] != "tree_branches" or any(key in cfg["variant"] for key in ["branch_length", "main_length", "tree_style", "branch_margin"]) or cfg["variant"].get("fork", "first") != "first":
        raise ValueError("This diagnostic currently supports full-length fixed fork-position trees")
    args.output.mkdir(parents=True, exist_ok=False)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("One GPU per precision diagnostic")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = os.environ["DFLASH_SOURCE"]
    if subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise RuntimeError("Source revision mismatch")
    subprocess.run(["git", "-C", source, "diff", "--exit-code", "HEAD", "--", "dflash/model.py"], check=True, capture_output=True)
    sys.path.insert(0, source)
    official = importlib.import_module("dflash.model")
    if sha(cfg["reference_rows"]) != cfg["reference_sha256"]:
        raise RuntimeError("Divergence reference changed")
    reference = [json.loads(line) for line in Path(cfg["reference_rows"]).read_text().splitlines()]
    originals = {r["problem_id"]: r for r in reference if r["method"] == "native" and r.get("repeat", 0) == 0}
    candidates = {r["problem_id"]: r for r in reference if r["method"] == "candidate" and r.get("repeat", 0) == 0}
    selected = []
    for problem, baseline in originals.items():
        other = candidates[problem]
        difference = first_difference(baseline["tokens"], other["tokens"])
        if difference is not None and 0 < difference < min(len(baseline["tokens"]), len(other["tokens"])):
            selected.append((problem, difference))
    selected = selected[:cfg["requests"]]
    if len(selected) != cfg["requests"]:
        raise RuntimeError("Not enough token-divergent reference requests")
    load = dict(cache_dir=os.environ["HF_HUB_CACHE"], local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")
    target = AutoModelForCausalLM.from_pretrained(TARGET["id"], revision=TARGET["revision"], **load).cuda().eval().requires_grad_(False)
    native = official.DFlashDraftModel.from_pretrained(DRAFT["id"], revision=DRAFT["revision"], **load).cuda().eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(TARGET["id"], revision=TARGET["revision"], cache_dir=os.environ["HF_HUB_CACHE"], local_files_only=True)
    eos = target.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    manifest = {r["problem_id"]: r for r in json.loads(Path(cfg["eval_manifest"]).read_text())["records"]}
    results = []
    for precision in ["bf16", "fp32"]:
        if precision == "fp32":
            cast_parameters(target, torch.float32)
            cast_parameters(native, torch.float32)
            torch.cuda.empty_cache()
        for problem, difference in selected:
            record = manifest[problem]
            prompt = record.get("prompt") or record["turns"][0]
            ids = torch.tensor(token_ids(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=False)), device="cuda").unsqueeze(0)
            arms, captured = {}, {}
            for method in ["native", "candidate"]:
                events = []
                wanted = ids.shape[1]+difference-1
                def hook(module, inputs, kwargs, output):
                    positions = kwargs["position_ids"][0]
                    event = {"positions": positions.cpu(), "input_ids": (inputs[0] if inputs else kwargs["input_ids"])[0].cpu()}
                    indices = (positions == wanted).nonzero().flatten()
                    event["rows"] = {int(i): {"logits": output.logits[0, i].detach().float().cpu(), "hidden": output.hidden_states[-1][0, i].detach().float().cpu()} for i in indices}
                    events.append(event)
                handle = target.register_forward_hook(hook, with_kwargs=True) if precision == "bf16" else None
                try:
                    generated = official.dflash_generate(native, target, ids, cfg["output_cap"], eos, 0., block_size=16, return_stats=True) if method == "native" else decode_variant(official, native, target, ids, cfg["output_cap"], eos, cfg["variant"])
                finally:
                    if handle is not None:
                        handle.remove()
                tokens = generated.output_ids[0, ids.shape[1]:].tolist()
                arms[method] = {"tokens": tokens, "acceptance_lengths": generated.acceptance_lengths}
                if precision == "bf16":
                    expected = originals[problem] if method == "native" else candidates[problem]
                    if tokens != expected["tokens"][:cfg["output_cap"]]:
                        raise RuntimeError("BF16 replay failed to reproduce saved outputs")
                    step, start, offset = prediction_step(generated.acceptance_lengths, difference)
                    event = events[step+1]  # target prefill is event zero
                    if method == "native":
                        node, prefix_nodes = offset, torch.arange(offset+1)
                    else:
                        branches = cfg["variant"]["branches"]
                        _, paths, _, _ = tree_layout(16, tuple(range(1, branches)), (16,)*branches)
                        path = paths[generated.trace[step]["winner"]]
                        node, prefix_nodes = int(path[offset]), path[:offset+1]
                    if event["input_ids"][prefix_nodes].tolist() != tokens[start:difference]:
                        raise RuntimeError("Selected target predictor was conditioned on a different token prefix")
                    values = event["rows"][node]
                    if int(values["logits"].argmax()) != tokens[difference]:
                        raise RuntimeError("Captured target argmax does not explain emitted token")
                    captured[method] = {**values, "event": step, "relative_node": node, "verified_input_tokens": len(event["input_ids"])}
            row = {"problem_id": problem, "benchmark": record["benchmark"], "precision": precision, "reference_first_difference": difference, "first_difference": first_difference(arms["native"]["tokens"], arms["candidate"]["tokens"]), "exact": arms["native"]["tokens"] == arms["candidate"]["tokens"], "arms": arms}
            if precision == "bf16":
                a, b = captured["native"], captured["candidate"]
                row["matching_prefix_probe"] = {"status": "pass", "logit_max_abs_difference": float((a["logits"]-b["logits"]).abs().max()), "hidden_relative_l2": float((a["hidden"]-b["hidden"]).norm()/a["hidden"].norm()), "arms": {}}
                for method, values in captured.items():
                    top = values["logits"].topk(5)
                    row["matching_prefix_probe"]["arms"][method] = {"argmax_token": int(values["logits"].argmax()), "top_ties": int((values["logits"] == values["logits"].max()).sum()), "top_token_ids": top.indices.tolist(), "top_logits": top.values.tolist(), "margin": float(top.values[0]-top.values[1]), "event": values["event"], "relative_node": values["relative_node"], "verified_input_tokens": values["verified_input_tokens"]}
            results.append(row)
            with (args.output/"diagnostics.jsonl").open("a") as stream:
                stream.write(json.dumps(row)+"\n")
            print(json.dumps({"precision": precision, "problem": problem, "exact": row["exact"], "first_difference": row["first_difference"]}), flush=True)
    (args.output/"precision-result.json").write_text(json.dumps({"status": "pass", "config": cfg, "target": TARGET, "draft": DRAFT, "source_commit": COMMIT, "source_sha256": sha(Path(source)/"dflash/model.py"), "rows": results, "scope": "Selected divergence diagnostics, not an accuracy-rate estimate. FP32 uses the same checkpoint values, parameter-only casting and TF32 disabled. No FP32 latency is used for speed claims. Token-prefix checks and higher-precision replay probe whether BF16 shape/cache-grouping perturbations explain disagreement."}, indent=2))


if __name__ == "__main__":
    main()
