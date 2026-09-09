"""Frozen RelaySpec hardware study; no training, tuning, or decoder edits."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source"))
from telemetry import Monitor, capture

MODELS = {
    "target": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "source": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
    "draft": ("z-lab/Qwen3-4B-DFlash-b16", "b74e3a329c4d963783143b1e970d95b002be72bd"),
    "native": ("z-lab/Qwen3-8B-DFlash-b16", "9b41424b7109f9c5413454f481b09a82b85333f4"),
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "main", "profile", "memory", "counters"], required=True)
    parser.add_argument("--method", choices=["all", "ar", "native", "source", "relay"], default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    assert sha(args.checkpoint) == protocol["checkpoint_sha256"]
    assert sha(HERE / "manifest.json") == protocol["manifest_sha256"]
    for name, expected in protocol["source_files"].items():
        assert sha(HERE / name) == expected, name
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "executed-run.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (args.output / "invocation.json").write_text(json.dumps(vars(args), default=str, indent=2))
    with Monitor(args.output / "telemetry.jsonl", 1.):
        result = run(args, protocol)
    (args.output / "complete.json").write_text(json.dumps(result, indent=2))


def run(args, protocol):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from relayspec.dflash import import_official_dflash
    from relayspec.mapper_campaign import restore_mapper
    from relayspec.source import SourceTapProvider
    from relayspec.benchmarking import benchmark_turns
    from relayspec.generation import native_autoregressive_generate, matched_full_target_dflash_generate, relay_dflash_generate
    assert transformers.__version__ == "5.3.0"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.manual_seed(protocol["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    official_source = Path(os.environ["DFLASH_SOURCE"])
    assert sha(official_source / "dflash/model.py") == protocol["official_model_sha256"]
    Draft, official_generate = import_official_dflash(official_source, protocol["official_commit"])
    cache = os.environ["HF_HUB_CACHE"]
    def load(key, cls=AutoModelForCausalLM):
        identifier, revision = MODELS[key]
        selected_cache = os.environ["SOURCE_HF_CACHE"] if key == "source" else cache
        model = cls.from_pretrained(identifier, revision=revision, cache_dir=selected_cache,
                                   local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")
        return model.cuda().eval().requires_grad_(False)
    methods = protocol["methods"] if args.method == "all" else [args.method]
    if args.stage == "memory":
        assert methods in [["source"], ["relay"]]
    target = load("target")
    source = load("source") if set(methods) & {"source", "relay"} else None
    # Isolated deployment excludes the two source blocks that the provider
    # never executes. Main timings retain the original co-resident protocol.
    if args.stage == "memory" and source is not None and "source" in methods:
        source.model.layers = torch.nn.ModuleList(list(source.model.layers[:34]))
        gc.collect(); torch.cuda.empty_cache()
    embedding = source.model.embed_tokens if source is not None else None
    head = source.lm_head if source is not None else None
    provider = None
    if "source" in methods:
        provider = SourceTapProvider(source, source_layers=34, tap_layers=(1, 9, 17, 25, 33)).to(device="cuda", dtype=torch.bfloat16).eval()
    if source is not None and "source" not in methods:
        del source
        source = None
        gc.collect(); torch.cuda.empty_cache()
    draft = load("draft", Draft) if set(methods) & {"source", "relay"} else None
    native = load("native", Draft) if "native" in methods else None
    mapper = None
    taps = ()
    if "relay" in methods:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        mapper, taps = restore_mapper(checkpoint, target_hidden_size=target.config.hidden_size,
                                      draft_hidden_size=draft.config.hidden_size, eps=target.config.rms_norm_eps)
        mapper = mapper.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
        del checkpoint
    tokenizer = AutoTokenizer.from_pretrained(*[MODELS["target"][0]], revision=MODELS["target"][1], cache_dir=cache, local_files_only=True)
    records = json.loads((HERE / "manifest.json").read_text())["records"]
    if args.stage != "main":
        records = [next(r for r in records if r["benchmark"] == b) for b in ["gsm8k", "math500", "humaneval", "mtbench"]]
    if args.stage in {"profile", "counters"}:
        records = [next(r for r in records if r["benchmark"] == "math500")]
    cap = {"smoke": 128, "profile": 256, "counters": 64}.get(args.stage, 2048)
    repeats = protocol["repeats"] if args.stage == "main" else 1
    eos = [tokenizer.eos_token_id]
    encoded = {}
    for record in records:
        prompt = benchmark_turns(record)[0]
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        encoded[record["problem_id"]] = (ids["input_ids"] if hasattr(ids, "keys") else ids).cuda()
    profiling = args.stage in {"profile", "counters"}
    class Regions:
        def __init__(self, method):
            self.method = method
        def region(self, name):
            return torch.cuda.nvtx.range(self.method + "::" + name)
    def generate(method, ids, length, annotate=False):
        common = dict(input_ids=ids, max_new_tokens=length, stop_token_ids=eos, temperature=0., return_stats=True)
        if method == "ar":
            return native_autoregressive_generate(target, **common)
        if method == "native":
            return official_generate(native, target=target, block_size=16, **common)
        common["profile_recorder"] = Regions(method) if annotate else None
        if method == "source":
            return matched_full_target_dflash_generate(draft, source_provider=provider, native_target=target, block_size=16, **common)
        return relay_dflash_generate(draft, relay=mapper, relay_target_layer_ids=taps, native_target=target,
                                    source_embedding=embedding, source_lm_head=head, block_size=16, **common)
    probe = encoded[records[0]["problem_id"]]
    expected = {}
    for method in methods:
        for _ in range(2):
            value = generate(method, probe, 64)
            del value
        # Repeat and instrumentation identity check on a short probe.
        a, b = generate(method, probe, 64), generate(method, probe, 64, annotate=profiling)
        assert torch.equal(a.output_ids, b.output_ids) and a.acceptance_lengths == b.acceptance_lengths, method
        del a, b
        if profiling:
            a = generate(method, probe, cap)
            expected[method] = (a.output_ids.clone(), list(a.acceptance_lengths))
            del a
    gc.collect(); torch.cuda.empty_cache()
    provenance = dict(stage=args.stage, methods=methods, cap=cap, repeats=repeats, requests=len(records),
        models=MODELS, checkpoint_sha256=protocol["checkpoint_sha256"], taps=taps, torch=torch.__version__,
        transformers=transformers.__version__, cuda=torch.version.cuda, hostname=platform.node(), gpu=torch.cuda.get_device_name(),
        source_transformer_resident=provider is not None, gate="short repeat and annotation identity passed",
        retained_source_blocks=len(source.model.layers) if source is not None else 0,
        memory_scope="isolated source/relay deployment" if args.stage == "memory" else "All evaluated arms co-resident; use separate memory runs for deployment footprints",
        script_sha256=sha(args.output / "executed-run.py"), protocol_sha256=sha(HERE / "protocol.json"),
        timing="synchronized wall time includes prefill and decoding; no CUDA events or profiler in throughput runs",
        nvidia_smi=capture(["nvidia-smi", "-q"]),
        parameter_bytes={name: sum(p.numel()*p.element_size() for p in model.parameters()) for name, model in
                         [("target", target), ("source", source), ("source_draft", draft), ("native_draft", native), ("mapper", mapper)] if model is not None})
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (args.output / "records.json").write_text(json.dumps(records, indent=2))
    target_original = target.forward
    counter = [0]
    current_method = [""]
    if profiling:
        def target_forward(*a, **kw):
            label = "target_prefill" if counter[0] == 0 else "target_decode"
            counter[0] += 1
            with torch.cuda.nvtx.range(current_method[0] + "::" + label):
                return target_original(*a, **kw)
        target.forward = target_forward
        torch.cuda.synchronize(); torch.cuda.profiler.start()
    rows = []
    try:
        with (args.output / "evaluation.jsonl").open("x") as stream:
            for repeat in range(repeats):
                for i, record in enumerate(records):
                    offset = (i + repeat) % len(methods)
                    for method in methods[offset:] + methods[:offset]:
                        ids = encoded[record["problem_id"]]
                        counter[0] = 0
                        current_method[0] = method
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                        before = torch.cuda.memory_allocated()
                        with torch.cuda.nvtx.range(method) if profiling else nullcontext():
                            start = time.monotonic()
                            value = generate(method, ids, cap, annotate=profiling)
                            torch.cuda.synchronize()
                            stop = time.monotonic()
                        tokens = value.output_ids[0, ids.shape[1]:].tolist()
                        if profiling:
                            wanted, lengths = expected[method]
                            assert torch.equal(value.output_ids, wanted) and value.acceptance_lengths == lengths
                        row = dict(method=method, repeat=repeat, problem_id=record["problem_id"], benchmark=record["benchmark"],
                            monotonic_start=start, monotonic_stop=stop, seconds=stop-start, input_tokens=ids.shape[1],
                            output_tokens=len(tokens), tokens=tokens, completion=tokenizer.decode(tokens, skip_special_tokens=True),
                            acceptance_lengths=list(value.acceptance_lengths), capped=len(tokens) >= cap,
                            allocated_before=before, peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
                            incremental_peak_allocated=torch.cuda.max_memory_allocated()-before,
                            time_to_first_token=getattr(value, "time_to_first_token", None),
                            source_executed_tokens=getattr(value, "source_executed_tokens", 0))
                        assert 0 < row["output_tokens"] <= cap and row["seconds"] > 0
                        rows.append(row); stream.write(json.dumps(row) + "\n"); stream.flush()
                        print(json.dumps({k: row[k] for k in ["method", "repeat", "problem_id", "output_tokens", "seconds"]}), flush=True)
                        del value
    finally:
        if profiling:
            torch.cuda.profiler.stop()
            target.forward = target_original
    first = {(r["method"], r["problem_id"]): r for r in rows if r["repeat"] == 0}
    assert all(r["tokens"] == first[r["method"], r["problem_id"]]["tokens"] and
               r["acceptance_lengths"] == first[r["method"], r["problem_id"]]["acceptance_lengths"] for r in rows)
    return dict(status="pass", stage=args.stage, rows=len(rows), requests=len(records), repeats=repeats,
                profiling=profiling, repeat_identity="pass", source_transformer_resident=provider is not None)


if __name__ == "__main__":
    main()
