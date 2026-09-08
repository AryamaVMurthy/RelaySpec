"""Paired standalone checkpoint evaluation; supports task/block and timing sweeps."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import compact_student, token_ids
from run_lane import TARGET, DRAFT, COMMIT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one visible GPU")
    if sha(spec["checkpoint"]) != spec["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint hash mismatch")
    manifest_sha = sha(spec["eval_manifest"])
    if manifest_sha != spec["eval_manifest_sha256"]:
        raise RuntimeError("Evaluation manifest changed")
    if spec["phase"] not in {"development", "confirmation"}:
        raise ValueError("Specify evaluation exposure")
    if spec["phase"] == "confirmation":
        selection = json.loads(Path(spec["frozen_selection"]).read_text())
        if selection["checkpoint_sha256"] != spec["checkpoint_sha256"] or selection["manifest_sha256"] != manifest_sha or selection["block_size"] != spec["block_size"] or selection["native_block_size"] != spec.get("native_block_size", 16):
            raise RuntimeError("Confirmation differs from frozen selection")
    records = [r for r in json.loads(Path(spec["eval_manifest"]).read_text())["records"] if r["benchmark"] == spec["benchmark"]][spec.get("offset", 0):][:spec["requests"]]
    if len(records) != spec["requests"] or len({r["problem_id"] for r in records}) != len(records):
        raise RuntimeError("Evaluation coverage is incomplete or duplicated")
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = os.environ["DFLASH_SOURCE"]
    if subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise RuntimeError("Native source revision mismatch")
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
    eos = target.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    models = {"native": (native, spec.get("native_block_size", 16)), "student": (student, spec["block_size"])}
    if spec.get("include_ar"):
        # The pinned official block=1 path bypasses the drafter entirely.
        models["ar"] = (native, 1)
    def encode(record):
        prompt = record.get("prompt") or record["turns"][0]
        return torch.tensor(token_ids(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=False)), device="cuda").unsqueeze(0)
    def generate(model, ids, cap, block):
        return official.dflash_generate(model, target, ids, cap, eos, 0., block_size=block, return_stats=True)
    probe_ids = encode(records[0])
    live = generate(student, probe_ids, min(64, spec["output_cap"]), spec["block_size"])
    reload = generate(restored, probe_ids, min(64, spec["output_cap"]), spec["block_size"])
    if not torch.equal(live.output_ids, reload.output_ids) or live.acceptance_lengths != reload.acceptance_lengths:
        raise RuntimeError("Checkpoint reload did not reproduce decoding")
    del restored, checkpoint, live, reload
    torch.cuda.empty_cache()
    provenance = {"spec": spec, "training_config": cfg, "target": TARGET, "draft": DRAFT, "source_commit": COMMIT, "source_file_sha256": sha(Path(source)/"dflash/model.py"), "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(), "job": os.environ["SLURM_JOB_ID"], "reload_gate": "pass", "scope": "Rotated paired wall time includes prefill; all models resident. Greedy native verifier unchanged. AR, when enabled, uses the original implementation with block=1 and no drafter forward."}
    (args.output/"provenance.json").write_text(json.dumps(provenance, indent=2))
    for model, block in models.values():
        generate(model, probe_ids, 16, block)
    rows = []
    with (args.output/"evaluation.jsonl").open("w") as stream:
        for repeat in range(spec.get("repeats", 1)):
            for i, record in enumerate(records):
                ids = encode(record)
                names = list(models)
                rotate = (i+repeat) % len(names)
                names = names[rotate:]+names[:rotate]
                for name in names:
                    model, block = models[name]
                    torch.cuda.synchronize(); begin = time.perf_counter()
                    result = generate(model, ids, spec["output_cap"], block)
                    torch.cuda.synchronize(); seconds = time.perf_counter()-begin
                    tokens = result.output_ids[0, ids.shape[1]:].tolist()
                    row = {"repeat": repeat, "method": name, "problem_id": record["problem_id"], "benchmark": record["benchmark"], "seconds": seconds, "input_tokens": ids.shape[1], "output_tokens": len(tokens), "tokens": tokens, "acceptance_lengths": result.acceptance_lengths, "block_size": block, "capped": len(tokens) >= spec["output_cap"], "completion": tokenizer.decode(tokens, skip_special_tokens=True)}
                    rows.append(row)
                    stream.write(json.dumps(row)+"\n"); stream.flush()
                print(json.dumps({"event": "request_done", "repeat": repeat, "index": i, "elapsed": time.perf_counter()-started}), flush=True)
    baseline = {(r["repeat"], r["problem_id"]): r for r in rows if r["method"] == "native"}
    initial = {(r["method"], r["problem_id"]): r for r in rows if r["repeat"] == 0}
    for row in rows:
        if any(row[k] != initial[row["method"], row["problem_id"]][k] for k in ["tokens", "acceptance_lengths"]):
            raise RuntimeError("Repeated inference changed tokens or acceptance")
    summary = {}
    for name in models:
        arm = [r for r in rows if r["method"] == name]
        summary[name] = {"tps": sum(r["output_tokens"] for r in arm)/sum(r["seconds"] for r in arm), "progress": sum(sum(r["acceptance_lengths"]) for r in arm)/sum(len(r["acceptance_lengths"]) for r in arm), "exact_native": sum(r["tokens"] == baseline[r["repeat"], r["problem_id"]]["tokens"] for r in arm), "requests": len(records), "timed_generations": len(arm), "capped_generations": sum(r["capped"] for r in arm)}
    for point in summary.values():
        point["native_ratio"] = point["tps"]/summary["native"]["tps"]
    result = {"status": "pass", "summary": summary, "elapsed_seconds": time.perf_counter()-started, "repeats_exact": True}
    (args.output/"evaluation-result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
