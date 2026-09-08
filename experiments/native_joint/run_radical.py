"""Short inference-only hypothesis screens against unchanged released DFlash."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import token_ids, compact_student
from radical_decode import decode_variant
from run_lane import TARGET, DRAFT, COMMIT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    from confirmation_protocol import validate_protocol
    protocol = validate_protocol(spec, Path(__file__).resolve().parent, TARGET, DRAFT, COMMIT)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"evaluation-protocol.json").write_text(json.dumps(protocol, indent=2))
    if torch.cuda.device_count() != 1:
        raise RuntimeError("One GPU per screening lane")
    torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = spec.get("allow_bf16_reduction", True)
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
    manifest = json.loads(Path(spec["eval_manifest"]).read_text())
    records = [record for benchmark in spec["benchmarks"] for record in [r for r in manifest["records"] if r["benchmark"] == benchmark][spec.get("eval_offset", 0):spec.get("eval_offset", 0)+spec["requests_per_benchmark"]]]
    if len(records) != len(spec["benchmarks"])*spec["requests_per_benchmark"] or len({r["problem_id"] for r in records}) != len(records):
        raise RuntimeError("Incomplete screening coverage")
    eos = target.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    def encode(record):
        prompt = record.get("prompt") or record["turns"][0]
        return torch.tensor(token_ids(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=False)), device="cuda").unsqueeze(0)
    def original(ids, cap):
        return official.dflash_generate(native, target, ids, cap, eos, 0., block_size=16, return_stats=True)
    provenance = {"spec": spec, "target": TARGET, "draft": DRAFT, "source_commit": COMMIT, "source_file_sha256": sha(Path(source)/"dflash/model.py"), "eval_manifest_sha256": sha(spec["eval_manifest"]), "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(), "job": os.environ["SLURM_JOB_ID"], "scope": "Inference-only adaptive development. No weights updated. Every proposal is greedily verified by the frozen target. Numerical agreement with the original block16 decoder is measured, not assumed. All latency includes prefill and policy overhead."}
    provenance["ddtree_baseline"] = {"upstream": "https://github.com/liranringel/ddtree", "commit": "c96427a185677bf4133ed865dd1626a5041aef9b", "scope": "Adapted heap builder in shared pinned runtime; instrumentation removed; vectorized leaf-path acceptance and shared cache compaction."}
    provenance["allow_bf16_reduced_precision_reduction"] = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    (args.output/"provenance.json").write_text(json.dumps(provenance, indent=2))
    # A byte-identical full-block control checks the copied cache/verification path.
    for record in records:
        ids = encode(record)
        baseline = original(ids, 64)
        noop = decode_variant(official, native, target, ids, 64, eos, {"kind": "fixed", "block_size": 16})
        if not torch.equal(baseline.output_ids, noop.output_ids) or baseline.acceptance_lengths != noop.acceptance_lengths:
            raise RuntimeError("Unmodified custom decoder failed original-equivalence gate")
        if any(variant["kind"] in {"tree_branches", "ddtree", "adaptive_leaves"} for variant in spec["variants"]):
            tree_noop = decode_variant(official, native, target, ids, 64, eos, {"kind": "tree_branches", "block_size": 16, "branches": 1})
            if not torch.equal(baseline.output_ids, tree_noop.output_ids) or baseline.acceptance_lengths != tree_noop.acceptance_lengths:
                raise RuntimeError("Single-path packed tree failed original-equivalence gate")
    (args.output/"noop-gate.json").write_text(json.dumps({"status": "pass", "requests": len(records), "output_cap": 64}, indent=2))
    candidate, candidate_target = native, target
    if spec.get("candidate_checkpoint"):
        if spec.get("quantization"):
            raise ValueError("Keep checkpoint and quantization experiments separate")
        checkpoint_spec = spec["candidate_checkpoint"]
        if sha(checkpoint_spec["path"]) != checkpoint_spec["sha256"]:
            raise RuntimeError("Candidate checkpoint hash mismatch")
        checkpoint = torch.load(checkpoint_spec["path"], map_location="cpu", weights_only=True)
        cfg = checkpoint["config"]
        if cfg.get("midpoint") or cfg.get("conditioning_prefix"):
            raise ValueError("This runner supports ordinary compact draft checkpoints only")
        candidate = compact_student(native, cfg["taps"], cfg["layers"]).eval().requires_grad_(False)
        candidate.load_state_dict(checkpoint["state_dict"])
        restored = compact_student(native, cfg["taps"], cfg["layers"]).eval().requires_grad_(False)
        restored.load_state_dict(checkpoint["state_dict"])
        ids = encode(records[0])
        live = decode_variant(official, candidate, target, ids, 64, eos, spec["variants"][0])
        replay = decode_variant(official, restored, target, ids, 64, eos, spec["variants"][0])
        if not torch.equal(live.output_ids, replay.output_ids) or live.acceptance_lengths != replay.acceptance_lengths:
            raise RuntimeError("Compact candidate reload failed exact output/progress gate")
        provenance["candidate_checkpoint"] = {**checkpoint_spec, "training_config": cfg, "reload_gate": "pass"}
        provenance["scope"] = "Previously jointly trained compact interface/drafter, now tested with packed verification. Frozen original target. Native single-path and optional released-drafter tree references share the runtime and hardware."
        (args.output/"provenance.json").write_text(json.dumps(provenance, indent=2))
        del checkpoint, restored, live, replay
        torch.cuda.empty_cache()
    if spec.get("quantization"):
        from quantized_draft import build_candidate
        candidate, candidate_target, quantization = build_candidate(native, target, spec["quantization"], args.output, official, encode(records[0]), eos)
        provenance["quantization"] = quantization
        provenance["scope"] = "Native draft and optional draft-only head quantization. Frozen BF16 target verifies every token. No training updates; original native and target controls are preserved. Full latency includes prefill and quantized inference overhead; one-time packing/reload is reported separately."
        (args.output/"provenance.json").write_text(json.dumps(provenance, indent=2))
    outcomes = []
    for variant in spec["variants"]:
        start = time.perf_counter()
        directory = args.output/variant["name"]
        directory.mkdir()
        rows = []
        try:
            probe = encode(records[0])
            original(probe, 16)
            decode_variant(official, candidate, candidate_target, probe, 16, eos, variant)
            if spec.get("reference_variant"):
                decode_variant(official, native, target, probe, 16, eos, spec["reference_variant"])
            with (directory/"evaluation.jsonl").open("w") as stream:
                for repeat, i, record in [(repeat, i, record) for repeat in range(spec.get("repeats", 1)) for i, record in enumerate(records)]:
                    ids = encode(record)
                    names = ["native", "candidate"]+(["reference"] if spec.get("reference_variant") else [])
                    rotate = (i+repeat)%len(names)
                    names = names[rotate:]+names[:rotate]
                    for name in names:
                        torch.cuda.synchronize(); begin = time.perf_counter()
                        if name == "native":
                            result = original(ids, spec["output_cap"])
                        elif name == "reference":
                            result = decode_variant(official, native, target, ids, spec["output_cap"], eos, spec["reference_variant"])
                        else:
                            result = decode_variant(official, candidate, candidate_target, ids, spec["output_cap"], eos, variant)
                        torch.cuda.synchronize(); seconds = time.perf_counter()-begin
                        tokens = result.output_ids[0, ids.shape[1]:].tolist()
                        row = {"repeat": repeat, "method": name, "problem_id": record["problem_id"], "benchmark": record["benchmark"], "seconds": seconds, "output_tokens": len(tokens), "tokens": tokens, "capped": len(tokens) >= spec["output_cap"], "input_tokens": ids.shape[1], "completion": tokenizer.decode(tokens, skip_special_tokens=True), "acceptance_lengths": result.acceptance_lengths, "trace": getattr(result, "trace", None)}
                        rows.append(row)
                        stream.write(json.dumps(row)+"\n"); stream.flush()
            baselines = {(r["repeat"], r["problem_id"]): r for r in rows if r["method"] == "native"}
            first = {(r["method"], r["problem_id"]): r for r in rows if r["repeat"] == 0}
            if any(any(r[key] != first[r["method"], r["problem_id"]][key] for key in ["tokens", "acceptance_lengths"]) for r in rows):
                raise RuntimeError("Repeated inference changed output or acceptance")
            summary = {}
            for name in ["native", "candidate"]+(["reference"] if spec.get("reference_variant") else []):
                arm = [r for r in rows if r["method"] == name]
                summary[name] = {"tps": sum(r["output_tokens"] for r in arm)/sum(r["seconds"] for r in arm), "exact_native": sum(r["tokens"] == baselines[r["repeat"], r["problem_id"]]["tokens"] for r in arm), "requests": len(records), "timed_generations": len(arm), "progress": sum(sum(r["acceptance_lengths"]) for r in arm)/sum(len(r["acceptance_lengths"]) for r in arm)}
            result = {"status": "pass", "variant": variant, "summary": summary, "native_ratio": summary["candidate"]["tps"]/summary["native"]["tps"], "elapsed_seconds": time.perf_counter()-start}
            if "reference" in summary:
                result["reference_ratio"] = summary["candidate"]["tps"]/summary["reference"]["tps"]
        except Exception as error:
            result = {"status": "failed", "variant": variant, "error": str(error), "traceback": traceback.format_exc(), "elapsed_seconds": time.perf_counter()-start}
            (directory/"result.json").write_text(json.dumps(result, indent=2))
            # Preserve failing hypotheses, but do not recover from unsafe CUDA state.
            if "CUDA" in str(error) or "out of memory" in str(error):
                raise
        (directory/"result.json").write_text(json.dumps(result, indent=2))
        outcomes.append(result)
        print(json.dumps(result), flush=True)
    (args.output/"radical-result.json").write_text(json.dumps(outcomes, indent=2))


if __name__ == "__main__":
    main()
