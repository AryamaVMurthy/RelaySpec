"""Short, paired native-compute exploration on one allocated GPU."""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from relayspec.ar_paper_evidence import summarize
from relayspec.benchmarking import benchmark_turns, rotate_methods
from relayspec.dflash import import_official_dflash
from relayspec.generation import native_autoregressive_generate
from relayspec.native_compute import FullLinear, fit_linear_mlp, fit_pruned_mlp, native_intervention


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    spec = json.loads(args.spec.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    storage = Path(os.environ["RELAYSPEC_CACHE_DIR"]) / "relayspec/native-compute" / os.environ["SLURM_JOB_ID"] / args.output.name
    storage.mkdir(parents=True, exist_ok=False)
    checkpoint_index = {}
    if torch.cuda.device_count() != 1:
        raise ValueError("Each native research lane requires one allocated GPU")
    torch.manual_seed(spec["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = yaml.safe_load(Path(spec["config"]).read_text())
    klass, generate = import_official_dflash(os.environ["DFLASH_SOURCE"], config["proposer"]["source_commit"])
    load = dict(cache_dir=os.environ["TRANSFORMERS_CACHE"], local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")
    target = AutoModelForCausalLM.from_pretrained(config["target"]["id"], revision=config["target"]["revision"], **load).cuda().eval().requires_grad_(False)
    draft = klass.from_pretrained(config["proposer"]["id"], revision=config["proposer"]["revision"], **load).cuda().eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(config["target"]["id"], revision=config["target"]["revision"], cache_dir=os.environ["TRANSFORMERS_CACHE"], local_files_only=True)
    stop_ids = target.generation_config.eos_token_id
    stop_ids = [stop_ids] if isinstance(stop_ids, int) else stop_ids
    methods = {"native": {"kind": "native"}, "duplicate": {"kind": "native"}, **spec["candidates"]}
    manifest = Path(spec["manifest"])
    records = json.loads(manifest.read_text())["records"]
    records = [r for r in records if r["benchmark"] == spec.get("benchmark", "gsm8k")]
    records = records[spec.get("offset", 0):][:spec["requests"]]
    if len(records) != spec["requests"] or len({r["problem_id"] for r in records}) != len(records):
        raise ValueError("Invalid evaluation request coverage")
    provenance = {"spec": spec, "spec_sha256": digest(args.spec), "manifest_sha256": digest(manifest), "target": config["target"], "proposer": config["proposer"], "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "host": platform.node(), "job": os.environ["SLURM_JOB_ID"], "draft_layers": len(draft.layers), "draft_parameters": sum(p.numel() for p in draft.parameters()), "precision": "BF16 target and draft, SDPA, TF32 disabled", "scope": "Native same-target exploration. Development requests, capped outputs, one fit seed. No full-answer quality or exact-AR guarantee. All original weights remain resident during paired timing, so memory is not isolated deployment."}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps({"event": "loaded", "seconds": time.perf_counter()-started}), flush=True)

    def native(ids, cap):
        return generate(draft, target, input_ids=ids, max_new_tokens=cap, stop_token_ids=stop_ids, temperature=0.0, block_size=spec.get("block_size", 16), return_stats=True)

    bridges = {}
    context_ranks = sorted({m.get("rank", 0) for m in methods.values() if m["kind"] == "shared_context"})
    if context_ranks:
        weights = [getattr(layer.self_attn, name).weight.detach() for layer in draft.layers for name in ["k_proj", "v_proj"]]
        context_sizes = [w.shape[0] for w in weights]
        full_context = FullLinear(torch.cat(weights, dim=0))
        bridges["context", 0] = full_context, context_sizes
    if any(m["kind"] in {"linear_mlp", "pruned_mlp"} or (m["kind"] == "shared_context" and m.get("rank", 0)) for m in methods.values()):
        capture_start = time.perf_counter()
        cache = Path(spec["calibration_cache"])
        index = json.loads((cache / "cache-index.json").read_text())
        if index["status"] != "pass" or index["metadata"]["config"]["target"] != config["target"]:
            raise ValueError("Calibration cache target/provenance mismatch")
        entries = sorted((e for e in index["entries"] if e["split"] == "train"), key=lambda e: e["index"])[:spec["calibration_records"]]
        layer_ids = sorted({i for m in methods.values() for i in m.get("layers", [])})
        captured = {i: [[], []] for i in layer_ids}
        boundaries = {i: [0] for i in layer_ids}
        context_rows, context_boundaries = [], [0]
        handles = []
        per_record = {}
        for i in layer_ids:
            def hook(module, inputs, output, layer=i):
                remaining = 128 - per_record.get(layer, 0)
                if remaining > 0:
                    x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])[:remaining]
                    y = output.detach().reshape(-1, output.shape[-1])[:remaining]
                    captured[layer][0].append(x.cpu())
                    captured[layer][1].append(y.cpu())
                    per_record[layer] = per_record.get(layer, 0) + len(x)
            handles.append(draft.layers[i].mlp.register_forward_hook(hook))
        if context_ranks:
            def context_hook(module, args, kwargs):
                x = kwargs["target_hidden"].detach().reshape(-1, draft.config.hidden_size)
                count = min(16, len(x), 128-per_record.get("context", 0))
                if count > 0:
                    positions = torch.linspace(0, len(x)-1, count, device=x.device).long()
                    context_rows.append(x[positions].cpu())
                    per_record["context"] = per_record.get("context", 0)+count
            handles.append(draft.layers[0].register_forward_pre_hook(context_hook, with_kwargs=True))
        calibration = []
        try:
            for entry in entries:
                path = cache / entry["file"]
                if digest(path) != entry["sha256"]:
                    raise ValueError("Calibration record hash mismatch")
                payload = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
                ids = payload["input_ids"].cuda()
                per_record.clear()
                native(ids, spec.get("calibration_output_cap", 64))
                calibration.append({"entry": entry, "input_token_ids": ids.cpu().tolist()})
                for i in layer_ids:
                    boundaries[i].append(boundaries[i][-1] + per_record.get(i, 0))
                if context_ranks:
                    context_boundaries.append(context_boundaries[-1]+per_record.get("context", 0))
        finally:
            for handle in handles:
                handle.remove()
        capture_seconds = time.perf_counter() - capture_start
        fit_start = time.perf_counter()
        diagnostics = []
        for i in layer_ids:
            x, y = [torch.cat(values).cuda().float() for values in captured[i]]
            # Entire last calibration records are held out from fitting.
            nval = len(x) - boundaries[i][len(entries)-max(1, len(entries)//4)]
            if nval < 1 or len(x) - nval < 2:
                raise ValueError("Insufficient disjoint calibration records for fitting")
            for rank in sorted({m["rank"] for m in methods.values() if m["kind"] == "linear_mlp" and i in m["layers"]}):
                with torch.no_grad():
                    module, train_loss = fit_linear_mlp(x[:-nval], y[:-nval], rank)
                    prediction = module(x[-nval:].to(torch.bfloat16)).float()
                    val_loss = ((prediction-y[-nval:]).square().sum(-1)/y[-nval:].square().sum(-1).clamp_min(1e-8)).mean().item()
                bridges[i, rank] = module
                diagnostics.append({"layer": i, "rank": rank, "effective_rank": module.down.shape[1], "train_positions": len(x)-nval, "validation_positions": nval, "train_relative_mse": train_loss, "validation_relative_mse": val_loss})
                path = storage/f"bridge-layer{i}-rank{rank}.pt"
                torch.save(module.state_dict(), path)
                checkpoint_index[path.name] = {"path": str(path), "sha256": digest(path)}
            pruning = sorted({(m["fraction"], m["rank"]) for m in methods.values() if m["kind"] == "pruned_mlp" and i in m["layers"]})
            for fraction, rank in pruning:
                with torch.no_grad():
                    module, indices = fit_pruned_mlp(draft.layers[i].mlp, x[:-nval], y[:-nval], fraction, rank)
                    prediction = module(x[-nval:].to(torch.bfloat16)).float()
                    val_loss = ((prediction-y[-nval:]).square().sum(-1)/y[-nval:].square().sum(-1).clamp_min(1e-8)).mean().item()
                bridges[i, "pruned", fraction, rank] = module
                diagnostics.append({"layer": i, "kind": "pruned_mlp", "fraction": fraction, "rank": rank, "retained_neurons": indices.cpu().tolist(), "train_positions": len(x)-nval, "validation_positions": nval, "validation_relative_mse": val_loss})
                path = storage/f"pruned-layer{i}-fraction{fraction}-rank{rank}.pt"
                # Original nonlinear weights are already pinned: only save the
                # selected indices and learned correction, not model copies.
                torch.save({"indices": indices.cpu(), "correction": None if module.correction is None else module.correction.state_dict()}, path)
                checkpoint_index[path.name] = {"path": str(path), "sha256": digest(path)}
        if context_ranks:
            x = torch.cat(context_rows).cuda().float()
            with torch.no_grad():
                y = full_context(x.bfloat16()).float()
            nval = len(x)-context_boundaries[len(entries)-max(1, len(entries)//4)]
            if nval < 1 or len(x)-nval < 2:
                raise ValueError("Insufficient disjoint context calibration records")
            for rank in context_ranks:
                if not rank:
                    continue
                with torch.no_grad():
                    module, train_loss = fit_linear_mlp(x[:-nval], y[:-nval], rank)
                    pred = module(x[-nval:].bfloat16()).float()
                    val_loss = ((pred-y[-nval:]).square().sum(-1)/y[-nval:].square().sum(-1).clamp_min(1e-8)).mean().item()
                bridges["context", rank] = module, context_sizes
                diagnostics.append({"kind": "shared_context", "rank": rank, "effective_rank": module.down.shape[1], "train_positions": len(x)-nval, "validation_positions": nval, "train_relative_mse": train_loss, "validation_relative_mse": val_loss})
                path = storage/f"shared-context-rank{rank}.pt"
                torch.save(module.state_dict(), path)
                checkpoint_index[path.name] = {"path": str(path), "sha256": digest(path)}
        (args.output/"checkpoints.json").write_text(json.dumps(checkpoint_index, indent=2))
        (args.output/"fitting.json").write_text(json.dumps({"records": calibration, "capture_seconds": capture_seconds, "fit_seconds": time.perf_counter()-fit_start, "diagnostics": diagnostics}, indent=2))
        print(json.dumps({"event": "fitted", "capture_seconds": capture_seconds, "fit_seconds": time.perf_counter()-fit_start, "diagnostics": diagnostics}), flush=True)

    def encode(record):
        text = benchmark_turns(record)[0]
        ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        return (ids if isinstance(ids, torch.Tensor) else ids["input_ids"]).cuda()

    # Identical warmup protocol for every candidate before request timing.
    for name, intervention in methods.items():
        with native_intervention(draft, intervention, bridges):
            native(encode(records[0]), 16)
    rows = []
    with (args.output/"records.jsonl").open("x") as stream:
        for idx, record in enumerate(records):
            ids = encode(record)
            for name in rotate_methods(tuple(methods), idx):
                with native_intervention(draft, methods[name], bridges):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    stats = native(ids, spec["max_new_tokens"])
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-t0
                output = stats.output_ids[0, ids.shape[1]:].cpu()
                row = {"method": name, "benchmark": record["benchmark"], "problem_id": record["problem_id"], "repetition": 0, "turn_index": 0, "input_tokens": ids.shape[1], "output_tokens": len(output), "request_seconds": elapsed, "output_token_ids": output.tolist(), "output_hash": hashlib.sha256(output.to(torch.int32).numpy().tobytes()).hexdigest(), "acceptance_lengths": stats.acceptance_lengths, "acceptance_length": sum(stats.acceptance_lengths)/len(stats.acceptance_lengths), "target_calls": len(stats.acceptance_lengths), "draft_calls": len(stats.acceptance_lengths), "completion": tokenizer.decode(output, skip_special_tokens=True)}
                rows.append(row)
                stream.write(json.dumps(row)+"\n"); stream.flush()
            print(json.dumps({"event": "request_done", "index": idx, "seconds": time.perf_counter()-started}), flush=True)
    base = {r["problem_id"]: r for r in rows if r["method"] == "native"}
    duplicate = [r for r in rows if r["method"] == "duplicate"]
    if any(any(r[k] != base[r["problem_id"]][k] for k in ["output_token_ids", "acceptance_lengths"]) for r in duplicate):
        raise RuntimeError("Identical native control changed outputs or acceptance")
    result = {"status": "pass", "elapsed_seconds": time.perf_counter()-started, "summary": summarize(rows, reference="native"), "exact_sequences_vs_native": {name: sum(r["output_token_ids"] == base[r["problem_id"]]["output_token_ids"] for r in rows if r["method"] == name) for name in methods}}
    (args.output/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
