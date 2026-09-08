"""Recover native checkpoints with correct nonpersistent positional buffers."""
import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import compact_student, token_ids
from run_lane import TARGET, DRAFT, COMMIT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = json.loads(args.config.read_text())
    if sha(spec["checkpoint"]) != spec["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint hash mismatch")
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one GPU")
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = os.environ["DFLASH_SOURCE"]
    if subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise RuntimeError("Source mismatch")
    subprocess.run(["git", "-C", source, "diff", "--exit-code", "HEAD", "--", "dflash/model.py"], check=True, capture_output=True)
    sys.path.insert(0, source)
    official = importlib.import_module("dflash.model")
    load = dict(cache_dir=os.environ["HF_HUB_CACHE"], local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")
    target = AutoModelForCausalLM.from_pretrained(TARGET["id"], revision=TARGET["revision"], **load).cuda().eval().requires_grad_(False)
    native = official.DFlashDraftModel.from_pretrained(DRAFT["id"], revision=DRAFT["revision"], **load).cuda().eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(TARGET["id"], revision=TARGET["revision"], cache_dir=os.environ["HF_HUB_CACHE"], local_files_only=True)
    student = compact_student(native, cfg["taps"], cfg["layers"]).eval().requires_grad_(False)
    student.load_state_dict(checkpoint["state_dict"])
    restored = compact_student(native, cfg["taps"], cfg["layers"]).eval().requires_grad_(False)
    restored.load_state_dict(checkpoint["state_dict"])
    rounded = copy.deepcopy(student).float().bfloat16()
    buffers = {name: {"native_dtype": str(buffer.dtype), "rounded_dtype": str(dict(rounded.named_buffers())[name].dtype), "max_rounding_error": float((buffer.float()-dict(rounded.named_buffers())[name].float()).abs().max())} for name, buffer in student.named_buffers() if torch.is_floating_point(buffer)}
    assert any(r["max_rounding_error"] > 0 for r in buffers.values())
    models = {"native": native, "duplicate": native, "correct_student": student, "rounded_buffers": rounded}
    records = [r for r in json.loads(Path(cfg["eval_manifest"]).read_text())["records"] if r["benchmark"] == cfg["benchmark"]][cfg.get("eval_offset", 0):][:cfg["eval_requests"]]
    eos = target.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    def encode(record):
        prompt = record.get("prompt") or record["turns"][0]
        return torch.tensor(token_ids(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=False)), device="cuda").unsqueeze(0)
    def generate(model, record, cap):
        ids = encode(record)
        result = official.dflash_generate(model, target, ids, cap, eos, 0., block_size=cfg["block_size"], return_stats=True)
        return result, result.output_ids[0, ids.shape[1]:].tolist()
    probe, _ = generate(student, records[0], 64)
    reloaded, _ = generate(restored, records[0], 64)
    if not torch.equal(probe.output_ids, reloaded.output_ids) or probe.acceptance_lengths != reloaded.acceptance_lengths:
        raise RuntimeError("Correct-buffer checkpoint reproduction failed")
    del restored, checkpoint
    torch.cuda.empty_cache()
    for model in models.values():
        generate(model, records[0], 16)
    rows = []
    for i, record in enumerate(records):
        ids = encode(record)
        names = list(models)
        names = names[i%len(names):]+names[:i%len(names)]
        for name in names:
            torch.cuda.synchronize(); start = time.perf_counter()
            result = official.dflash_generate(models[name], target, ids, cfg["output_cap"], eos, 0., block_size=cfg["block_size"], return_stats=True)
            torch.cuda.synchronize(); seconds = time.perf_counter()-start
            tokens = result.output_ids[0, ids.shape[1]:].tolist()
            rows.append({"method": name, "problem_id": record["problem_id"], "benchmark": record["benchmark"], "seconds": seconds, "tokens": tokens, "output_tokens": len(tokens), "acceptance_lengths": result.acceptance_lengths, "capped": len(tokens) >= cfg["output_cap"]})
        print(json.dumps({"event": "request_done", "index": i}), flush=True)
    baseline = {r["problem_id"]: r for r in rows if r["method"] == "native"}
    summary = {}
    for name in models:
        arm = [r for r in rows if r["method"] == name]
        if name == "duplicate":
            assert all(all(r[k] == baseline[r["problem_id"]][k] for k in ["tokens", "acceptance_lengths"]) for r in arm)
        summary[name] = {"tps": sum(r["output_tokens"] for r in arm)/sum(r["seconds"] for r in arm), "progress": sum(sum(r["acceptance_lengths"]) for r in arm)/sum(len(r["acceptance_lengths"]) for r in arm), "exact_native": sum(r["tokens"] == baseline[r["problem_id"]]["tokens"] for r in arm), "requests": len(arm)}
    for point in summary.values():
        point["native_ratio"] = point["tps"]/summary["native"]["tps"]
    (args.output/"evaluation.json").write_text(json.dumps(rows, indent=2))
    result = {"status": "pass", "config": cfg, "recovery": spec, "buffers": buffers, "reload_gate": "pass", "summary": summary, "scope": "Correct student uses original FP32 nonpersistent RoPE buffers. Rounded buffer arm diagnoses the earlier casting issue. Single-seed development comparison, no independent AR guarantee."}
    (args.output/"recovery-result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
