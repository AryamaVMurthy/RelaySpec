"""Hardware replication of four frozen native methods; never fits or selects models."""
import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from telemetry import Monitor, capture


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    assert spec["purpose"] == "hardware_replication"
    manifest_path = HERE / spec["manifest"]
    assert sha(manifest_path) == spec["manifest_sha256"]
    assert sha(args.checkpoint) == spec["checkpoint_sha256"]
    records = json.loads(manifest_path.read_text())["records"]
    assert len(records) == spec["requests"] and len({r["problem_id"] for r in records}) == len(records)
    # The decoder under replication remains byte-identical to the L40S selection.
    frozen = json.loads((ROOT / "data/frozen-native-confirmation.json").read_text())
    for name, expected in frozen["protocol"]["local_source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(spec, indent=2))
    preflight = {"hostname": platform.node(), "machine": platform.machine(),
                 "python": sys.version, "nvidia_smi": capture(["nvidia-smi", "-q"]),
                 "processes": capture(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"]),
                 "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(HERE.glob("*.py"))},
                 "decoder_source_sha256": frozen["protocol"]["local_source_sha256"],
                 "config_sha256": sha(args.config), "manifest_sha256": sha(manifest_path)}
    (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2))
    with Monitor(args.output / "telemetry.jsonl", spec["telemetry_interval_seconds"]):
        completion = run(args, spec, records, preflight)
    (args.output / "complete.json").write_text(json.dumps(completion, indent=2))


def run(args, spec, records, preflight):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from core import compact_student, token_ids
    from radical_decode import decode_variant
    from run_lane import TARGET, DRAFT, COMMIT
    assert transformers.__version__ == spec["required_transformers"], "Install the specified Transformers version in an isolated environment"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    source = os.environ["DFLASH_SOURCE"]
    assert subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip() == COMMIT
    subprocess.run(["git", "-C", source, "diff", "--exit-code", "HEAD", "--", "dflash/model.py"], check=True, capture_output=True)
    assert sha(Path(source) / "dflash/model.py") == spec["official_model_sha256"]
    sys.path.insert(0, source)
    official = importlib.import_module("dflash.model")
    load = dict(cache_dir=os.environ["HF_HUB_CACHE"], local_files_only=True,
                dtype=torch.bfloat16, attn_implementation="sdpa")
    target = AutoModelForCausalLM.from_pretrained(TARGET["id"], revision=TARGET["revision"], **load).cuda().eval().requires_grad_(False)
    native = official.DFlashDraftModel.from_pretrained(DRAFT["id"], revision=DRAFT["revision"], **load).cuda().eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(TARGET["id"], revision=TARGET["revision"], cache_dir=os.environ["HF_HUB_CACHE"], local_files_only=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    assert cfg["taps"] == [25, 33] and cfg["layers"] == [0, 1, 2, 3, 4]
    compact = compact_student(native, cfg["taps"], cfg["layers"]).eval().requires_grad_(False)
    compact.load_state_dict(checkpoint["state_dict"], strict=True)
    eos = target.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    def encode(r):
        ids = tokenizer.apply_chat_template([{"role": "user", "content": r.get("prompt") or r["turns"][0]}],
                                           tokenize=True, add_generation_prompt=True, enable_thinking=False)
        return torch.tensor(token_ids(ids), device="cuda").unsqueeze(0)
    encoded = {r["problem_id"]: encode(r) for r in records}
    def generate(method, ids, cap):
        if method in {"native", "compact_linear", "ar"}:
            return official.dflash_generate(compact if method == "compact_linear" else native,
                target, ids, cap, eos, 0., block_size=1 if method == "ar" else 16, return_stats=True)
        return decode_variant(official, compact if method == "candidate" else native, target,
            ids, cap, eos, {"kind": "ddtree", "block_size": 16,
                           "tree_budget": 47 if method == "candidate" else 63})
    # Exact reload and no-op checks occur outside timed samples.
    probe = encoded[records[0]["problem_id"]]
    restored = compact_student(native, cfg["taps"], cfg["layers"]).eval().requires_grad_(False)
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    v = {"kind": "ddtree", "block_size": 16, "tree_budget": 47}
    a = generate("candidate", probe, 64)
    b = decode_variant(official, restored, target, probe, 64, eos, v)
    assert torch.equal(a.output_ids, b.output_ids) and a.acceptance_lengths == b.acceptance_lengths
    del checkpoint, restored, a, b
    gc.collect(); torch.cuda.empty_cache()
    for ids in encoded.values():
        a = generate("native", ids, 64)
        for variant in [{"kind": "fixed", "block_size": 16}, {"kind": "tree_branches", "block_size": 16, "branches": 1}]:
            b = decode_variant(official, native, target, ids, 64, eos, variant)
            assert torch.equal(a.output_ids, b.output_ids) and a.acceptance_lengths == b.acceptance_lengths
    del a, b
    (args.output / "gates.json").write_text(json.dumps({"reload": "pass", "native_noop": "pass", "single_path_tree": "pass", "requests": len(records)}))
    provenance = dict(preflight, target=TARGET, draft=DRAFT, source_commit=COMMIT,
        torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
        checkpoint_sha256=spec["checkpoint_sha256"], training_config=cfg,
        memory_scope="All three models resident; per-arm peaks are not isolated deployment footprints. Incremental allocated memory is measured separately.",
        comparison_scope="Hardware replication on previously evaluated requests; no fitting or tuning. Runtime differences from L40S must be reported.",
        resident_parameter_bytes={k: sum(p.numel()*p.element_size() for p in m.parameters()) for k, m in [("target", target), ("native", native), ("compact", compact)]})
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    methods = ["native", "candidate", "reference", "compact_linear"]
    for method in methods:
        for _ in range(2):
            result = generate(method, probe, 64)
            del result
    rows = []
    with (args.output / "evaluation.jsonl").open("x") as stream:
        for repeat in range(spec["repeats"]):
            for i, record in enumerate(records):
                ids = encoded[record["problem_id"]]
                offset = (i + repeat) % len(methods)
                for method in methods[offset:] + methods[:offset]:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    allocated = torch.cuda.memory_allocated()
                    reserved = torch.cuda.memory_reserved()
                    start = time.monotonic()
                    result = generate(method, ids, spec["output_cap"])
                    torch.cuda.synchronize()
                    stop = time.monotonic()
                    tokens = result.output_ids[0, ids.shape[1]:].tolist()
                    row = dict(method=method, repeat=repeat, problem_id=record["problem_id"], benchmark=record["benchmark"],
                        seconds=stop-start, monotonic_start=start, monotonic_stop=stop,
                        input_tokens=ids.shape[1], output_tokens=len(tokens), tokens=tokens,
                        capped=len(tokens) >= spec["output_cap"], completion=tokenizer.decode(tokens, skip_special_tokens=True),
                        acceptance_lengths=result.acceptance_lengths,
                        allocated_before=allocated, reserved_before=reserved,
                        peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
                    row["incremental_peak_allocated"] = row["peak_allocated"] - allocated
                    rows.append(row); stream.write(json.dumps(row) + "\n"); stream.flush()
                    print(json.dumps({k: row[k] for k in ["method", "repeat", "problem_id", "seconds", "output_tokens"]}), flush=True)
                    del result
    first = {(r["method"], r["problem_id"]): r for r in rows if r["repeat"] == 0}
    assert len(rows) == len(records) * 4 * spec["repeats"]
    assert all(all(r[k] == first[r["method"], r["problem_id"]][k] for k in ["tokens", "acceptance_lengths"]) for r in rows)
    if spec["include_ar_quality"]:
        with (args.output / "ar-quality.jsonl").open("x") as stream:
            for r in records:
                ids = encoded[r["problem_id"]]
                result = generate("ar", ids, spec["output_cap"])
                tokens = result.output_ids[0, ids.shape[1]:].tolist()
                stream.write(json.dumps(dict(method="ar", problem_id=r["problem_id"], benchmark=r["benchmark"],
                    tokens=tokens, output_tokens=len(tokens), capped=len(tokens) >= spec["output_cap"],
                    completion=tokenizer.decode(tokens, skip_special_tokens=True))) + "\n"); stream.flush()
                del result
    return {"status": "pass", "requests": len(records), "timed_rows": len(rows), "repeat_identity": "pass", "ar_quality": spec["include_ar_quality"]}


if __name__ == "__main__":
    main()
