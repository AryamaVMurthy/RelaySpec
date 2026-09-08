"""Attribute native generation time with CUDA events; diagnostic, not a speed claim."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import token_ids
from run_lane import TARGET, DRAFT, COMMIT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("One GPU per profile lane")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = os.environ["DFLASH_SOURCE"]
    if subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise RuntimeError("Source revision mismatch")
    subprocess.run(["git", "-C", source, "diff", "--exit-code", "HEAD", "--", "dflash/model.py"], check=True, capture_output=True)
    sys.path.insert(0, source)
    official = importlib.import_module("dflash.model")
    load = dict(cache_dir=os.environ["HF_HUB_CACHE"], local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")
    target = AutoModelForCausalLM.from_pretrained(TARGET["id"], revision=TARGET["revision"], **load).cuda().eval().requires_grad_(False)
    native = official.DFlashDraftModel.from_pretrained(DRAFT["id"], revision=DRAFT["revision"], **load).cuda().eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(TARGET["id"], revision=TARGET["revision"], cache_dir=os.environ["HF_HUB_CACHE"], local_files_only=True)
    eos = target.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    manifest = json.loads(Path(cfg["eval_manifest"]).read_text())
    records = [r for benchmark in cfg["benchmarks"] for r in [r for r in manifest["records"] if r["benchmark"] == benchmark][:cfg["requests_per_benchmark"]]]
    rows = []
    for record in records:
        prompt = record.get("prompt") or record["turns"][0]
        ids = torch.tensor(token_ids(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=False)), device="cuda").unsqueeze(0)
        def generate(cap):
            return official.dflash_generate(native, target, ids, cap, eos, 0., block_size=16, return_stats=True)
        generate(32)
        reference = generate(cfg["output_cap"])
        events, context = [], []
        calls = {"target": 0}
        saved = [(target, target.forward), (native, native.forward), (target.lm_head, target.lm_head.forward)]
        def wrapper(original, category):
            def forward(*args, **kwargs):
                name = category
                if category == "target":
                    name = "prefill" if calls["target"] == 0 else "verification"
                    calls["target"] += 1
                elif category == "head":
                    name = f"{context[-1]}_head" if context else "draft_head"
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                context.append(name)
                try:
                    return original(*args, **kwargs)
                finally:
                    context.pop()
                    end.record()
                    events.append((name, begin, end))
            return forward
        try:
            for (module, original), category in zip(saved, ["target", "draft", "head"]):
                module.forward = wrapper(original, category)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = generate(cfg["output_cap"])
            torch.cuda.synchronize()
            wall = time.perf_counter()-start
        finally:
            for module, original in saved:
                module.forward = original
        if not torch.equal(reference.output_ids, result.output_ids) or reference.acceptance_lengths != result.acceptance_lengths:
            raise RuntimeError("Profiling changed native output or acceptance")
        phases = {}
        for name, begin, end in events:
            phases.setdefault(name, {"seconds": 0., "calls": 0})
            phases[name]["seconds"] += begin.elapsed_time(end)/1000
            phases[name]["calls"] += 1
        row = {"benchmark": record["benchmark"], "problem_id": record["problem_id"], "input_tokens": ids.shape[1], "output_tokens": result.num_output_tokens, "wall_seconds": wall, "phases": phases, "output_gate": "pass", "acceptance_lengths": result.acceptance_lengths}
        rows.append(row)
        with (args.output/"profile.jsonl").open("a") as stream:
            stream.write(json.dumps(row)+"\n")
    result = {"status": "pass", "config": cfg, "target": TARGET, "draft": DRAFT, "source_commit": COMMIT, "source_file_sha256": sha(Path(source)/"dflash/model.py"), "manifest_sha256": sha(cfg["eval_manifest"]), "job": os.environ["SLURM_JOB_ID"], "gpu": torch.cuda.get_device_name(), "rows": rows, "scope": "CUDA event spans can include device idle time during host dispatch. Prefill/verification include their nested head spans, so do not sum those heads twice. Instrumented wall times are diagnostics, not candidate throughput evidence."}
    (args.output/"profile-result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": "pass", "profiled_requests": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
