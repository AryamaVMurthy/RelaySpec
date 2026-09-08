"""Independent native joint-training lane; no cross-model interface dependencies."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from functools import lru_cache
import random
import subprocess
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import TAPS, block_view, block_batch, compact_student, prediction_loss, token_ids, cast_parameters, conditioned_noise

TARGET = {"id": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218"}
DRAFT = {"id": "z-lab/Qwen3-8B-DFlash-b16", "revision": "9b41424b7109f9c5413454f481b09a82b85333f4"}
COMMIT = "94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    def log(event, **kw):
        print(json.dumps({"event": event, "elapsed": time.perf_counter()-started, **kw}), flush=True)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("One CUDA device must be visible per lane")
    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = os.environ["DFLASH_SOURCE"]
    actual = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        raise RuntimeError("Native DFlash source revision mismatch")
    subprocess.run(["git", "-C", source, "diff", "--exit-code", "HEAD", "--", "dflash/model.py"], check=True, capture_output=True)
    sys.path.insert(0, source)
    official = importlib.import_module("dflash.model")
    load = dict(cache_dir=os.environ["HF_HUB_CACHE"], local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")
    target = AutoModelForCausalLM.from_pretrained(TARGET["id"], revision=TARGET["revision"], **load).cuda().eval().requires_grad_(False)
    native = official.DFlashDraftModel.from_pretrained(DRAFT["id"], revision=DRAFT["revision"], **load).cuda().eval().requires_grad_(False)
    if native.target_layer_ids != TAPS:
        raise RuntimeError("Unexpected native feature taps")
    tokenizer = AutoTokenizer.from_pretrained(TARGET["id"], revision=TARGET["revision"], cache_dir=os.environ["HF_HUB_CACHE"], local_files_only=True)
    eos = target.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    scratch = Path(os.environ["NATIVE_SCRATCH"])/os.environ["SLURM_JOB_ID"]/args.output.name
    scratch.mkdir(parents=True, exist_ok=False)
    provenance = {"config": cfg, "target": TARGET, "draft": DRAFT, "source_commit": actual, "source_file_sha256": sha(Path(source)/"dflash/model.py"), "eval_manifest_sha256": sha(cfg["eval_manifest"]), "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(), "job": os.environ["SLURM_JOB_ID"], "scope": "Separate native compression/joint fine-tuning study. Target, embedding and head frozen. Student initialized from native same-target weights. Paired wall time includes prefill. All models remain resident during evaluation; memory is not isolated deployment."}
    (args.output/"provenance.json").write_text(json.dumps(provenance, indent=2))
    log("loaded")

    # Cache the frozen target once for each training sequence. Only raw corpus
    # records are read; all activations are created for this separate experiment.
    cache_manifest, datasets, seen = [], {}, set()
    if cfg.get("prepared_index"):
        prepared_path = Path(cfg["prepared_index"])
        prepared = json.loads(prepared_path.read_text())
        if prepared["status"] != "pass" or prepared["target"] != TARGET or prepared["sequence_length"] != cfg["sequence_length"]:
            raise RuntimeError("Prepared feature cache provenance mismatch")
        for split, count in [("train", cfg["train_records"]), ("validation", cfg["val_records"])]:
            selected = sorted([e for e in prepared["entries"] if e["split"] == split and e["tokens"]-e["anchor_min"] >= cfg["block_size"]], key=lambda e: e["index"])[:count]
            if len(selected) != count:
                raise RuntimeError("Insufficient prepared records")
            datasets[split] = []
            for entry in selected:
                if entry["question_sha"] in seen:
                    raise RuntimeError("Prepared training/validation overlap")
                seen.add(entry["question_sha"])
                datasets[split].append(entry["path"])
                cache_manifest.append(entry)
        (args.output/"prepared-source.json").write_text(json.dumps({"path": str(prepared_path), "sha256": sha(prepared_path), "source_job": prepared["source_job"]}, indent=2))
        log("prepared_cache", train=len(datasets["train"]), validation=len(datasets["validation"]))
    else:
        for split, count in [("train", cfg["train_records"]), ("validation", cfg["val_records"])]:
            path = Path(cfg["data_root"])/("train-32768.json" if split == "train" else "validation.json")
            manifest = json.loads(path.read_text())
            datasets[split] = []
            rejected = []
            for index, record in enumerate(manifest["records"]):
                if index % cfg.get("data_shards", 1) != cfg.get("data_shard", 0):
                    continue
                key = record["normalized_problem_sha256"]
                if key in seen:
                    raise RuntimeError("Training/validation question overlap")
                prompt = token_ids(tokenizer.apply_chat_template([{"role": "user", "content": record["problem"]}], tokenize=True, add_generation_prompt=True, enable_thinking=False))
                ids = prompt + tokenizer.encode(record["solution"], add_special_tokens=False)
                ids = ids[:cfg["sequence_length"]]
                lo, hi = len(prompt), len(ids)-cfg["block_size"]
                if lo > hi:
                    rejected.append({"index": index, "reason": "No complete answer block within sequence cap", "question_sha": key})
                    continue
                seen.add(key)
                ids = torch.tensor(ids, dtype=torch.long)
                with torch.no_grad():
                    output = target.model(input_ids=ids.unsqueeze(0).cuda(), use_cache=False, output_hidden_states=True)
                    taps = torch.stack([output.hidden_states[i+1][0] for i in TAPS]).cpu()
                    final_hidden = output.last_hidden_state[0].cpu()
                    if split == "train" and not datasets[split]:
                        altered = ids.clone()
                        altered[lo+1:] = 0
                        probe = target.model(input_ids=altered.unsqueeze(0).cuda(), use_cache=False, output_hidden_states=True)
                        causal = torch.equal(probe.last_hidden_state[0, :lo+1].cpu(), final_hidden[:lo+1]) and all(torch.equal(probe.hidden_states[t+1][0, :lo].cpu(), taps[j, :lo]) for j, t in enumerate(TAPS))
                        if not causal:
                            raise RuntimeError("Future tokens affected prefix target features")
                        (args.output/"causality-gate.json").write_text(json.dumps({"status": "pass", "anchor": lo, "tokens": len(ids), "test": "Alter all tokens after the anchor at fixed tensor shape; compare all conditioning taps and target next-token hidden state exactly."}, indent=2))
                        del probe
                payload = {"input_ids": ids, "taps": taps, "final_hidden": final_hidden, "anchor_min": lo, "anchor_max": hi, "question_sha": key}
                cache_path = scratch/f"{split}-{index:06}.pt"
                torch.save(payload, cache_path)
                datasets[split].append(str(cache_path))
                cache_manifest.append({"split": split, "index": index, "question_sha": key, "path": str(cache_path), "sha256": sha(cache_path), "tokens": len(ids), "input_ids": ids.tolist(), "anchor_min": lo, "anchor_max": hi})
                del output
                if len(datasets[split]) == count:
                    break
            if len(datasets[split]) != count:
                raise RuntimeError("Not enough eligible sequences")
            (args.output/f"{split}-data.json").write_text(json.dumps({"source": str(path), "manifest_sha256": sha(path), "provenance": manifest["provenance"], "eligible_records": count, "excluded": rejected}, indent=2))
            log("cache_split", split=split, records=count)
    (args.output/"cache.json").write_text(json.dumps(cache_manifest, indent=2))
    if cfg.get("prepare_only"):
        result = {"status": "prepared", "train_records": len(datasets["train"]), "validation_records": len(datasets["validation"]), "elapsed_seconds": time.perf_counter()-started}
        (args.output/"prepared-result.json").write_text(json.dumps(result, indent=2))
        log("prepared", summary=result)
        return
    expected_cache = {entry["path"]: entry for entry in cache_manifest}
    verified_cache = set()
    @lru_cache(maxsize=8)
    def load_record(path):
        if path not in verified_cache:
            if sha(path) != expected_cache[path]["sha256"]:
                raise RuntimeError("Prepared feature cache hash mismatch")
            verified_cache.add(path)
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record["question_sha"] != expected_cache[path]["question_sha"] or len(record["input_ids"]) != expected_cache[path]["tokens"]:
            raise RuntimeError("Prepared feature cache content mismatch")
        record["anchor_max"] = len(record["input_ids"])-cfg["block_size"]
        return record
    student = compact_student(native, cfg["taps"], cfg["layers"]).eval()
    batch_size = cfg.get("batch_size", 1)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if batch_size > 1:
        # Compare real native/student forwards against independent unpadded calls.
        # This gate runs before training and also perturbs every padding feature.
        with torch.no_grad():
            probe_records = [load_record(path) for path in datasets["train"][:2]]
            if len(probe_records) != 2:
                raise ValueError("Batched validation requires at least two records")
            probe_anchors = [probe_records[0]["anchor_min"], probe_records[1]["anchor_max"]]
            if probe_anchors[0] == probe_anchors[1]:
                probe_anchors[0] = max(1, probe_anchors[0]-1)
            gate = []
            probes = [(model, taps, name, dtype) for model, taps, name in [(native, TAPS, "native"), (student, cfg["taps"], "student")] for dtype in [torch.bfloat16, torch.float32]]
            for model, taps, name, dtype in probes:
                cast_parameters(model, dtype)
                view = block_batch(probe_records, probe_anchors, taps, cfg["block_size"])
                tokens = torch.full((2, cfg["block_size"]), native.mask_token_id, dtype=torch.long, device="cuda")
                tokens[:, 0] = view["anchor_token"].cuda()
                noise = target.model.embed_tokens(tokens).to(dtype)
                context = view["context"].cuda().to(dtype)
                kwargs = dict(noise_embedding=noise, position_ids=view["positions"].cuda(), attention_mask=view["attention_mask"].cuda(), use_cache=False, is_causal=False)
                batched = model(target_hidden=context, **kwargs)
                individual = []
                for i, (record, anchor) in enumerate(zip(probe_records, probe_anchors)):
                    single = block_view(record, anchor, taps, cfg["block_size"])
                    individual.append(model(target_hidden=single["context"].cuda().to(dtype), noise_embedding=noise[i:i+1], position_ids=single["positions"].cuda(), use_cache=False, is_causal=False))
                individual = torch.cat(individual)
                relative_error = float((batched.float()-individual.float()).norm()/individual.float().norm())
                max_error = float((batched.float()-individual.float()).abs().max())
                output_scale = float(individual.float().abs().max())
                for i, anchor in enumerate(probe_anchors):
                    context[i, anchor:] = 100
                perturbed = model(target_hidden=context, **kwargs)
                padding_exact = torch.equal(batched, perturbed)
                relative_limit, max_scaled_limit = ((.01, .02) if dtype == torch.bfloat16 else (1e-5, 1e-5))
                passed = relative_error <= relative_limit and max_error <= max_scaled_limit*output_scale and padding_exact
                gate.append({"model": name, "dtype": str(dtype), "relative_l2": relative_error, "relative_l2_limit": relative_limit, "max_absolute": max_error, "output_max_absolute": output_scale, "max_scaled_limit": max_scaled_limit, "padding_perturbation_exact": padding_exact, "passed": passed})
                (args.output/"batch-gate.json").write_text(json.dumps({"status": "running" if passed else "failed", "anchors": probe_anchors, "models": gate}, indent=2))
                if not passed:
                    raise RuntimeError(f"Batched forward gate failed: {gate[-1]}")
                cast_parameters(model, torch.bfloat16)
            (args.output/"batch-gate.json").write_text(json.dumps({"status": "pass", "anchors": probe_anchors, "models": gate}, indent=2))
            del batched, individual, perturbed, context, noise, kwargs, probe_records
        log("batch_gate", models=gate)
    frozen_pointers = {p.data_ptr() for model in [native, target] for p in model.parameters()}
    if any(p.data_ptr() in frozen_pointers for p in student.parameters()):
        raise RuntimeError("Student shares parameter storage with a frozen model")
    models = {"native": native, "duplicate": native, "student": student}
    eval_manifest = json.loads(Path(cfg["eval_manifest"]).read_text())
    if cfg.get("benchmarks"):
        evaluation = [r for benchmark in cfg["benchmarks"] for r in [r for r in eval_manifest["records"] if r["benchmark"] == benchmark][cfg.get("eval_offset", 0):][:cfg["requests_per_benchmark"]]]
    else:
        evaluation = [r for r in eval_manifest["records"] if r["benchmark"] == cfg["benchmark"]][cfg.get("eval_offset", 0):][:cfg["eval_requests"]]
    if len(evaluation) != cfg["eval_requests"]:
        raise RuntimeError("Incomplete evaluation coverage")
    def encode(record):
        prompt = record.get("prompt") or record["turns"][0]
        return torch.tensor(token_ids(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=False)), device="cuda").unsqueeze(0)

    def generate(model, ids, cap):
        if cfg.get("conditioning_prefix") and model is not native:
            from radical_decode import decode_variant
            return decode_variant(official, native, target, ids, cap, eos, {"kind": "refine", "block_size": cfg["block_size"], "prefix": cfg["conditioning_prefix"], "threshold": cfg.get("refine_threshold", 0.)}, refiner=model)
        return official.dflash_generate(model, target, ids, cap, eos, 0., block_size=cfg["block_size"], return_stats=True)

    midpoint = None
    if cfg.get("midpoint"):
        from midpoint_conditioning import MidpointConditioning
        if cfg.get("conditioning_prefix"):
            raise ValueError("Midpoint and second-pass refinement are separate protocols")
        probe = encode(evaluation[0])
        plain = generate(student, probe, 64)
        midpoint = MidpointConditioning(student, target, cfg["midpoint"])
        strength, midpoint.strength = midpoint.strength, 0.
        zero = generate(student, probe, 64)
        if not torch.equal(plain.output_ids, zero.output_ids) or plain.acceptance_lengths != zero.acceptance_lengths:
            raise RuntimeError("Zero midpoint injection changed native architecture")
        midpoint.strength = strength
        (args.output/"midpoint-zero-gate.json").write_text(json.dumps({"status": "pass", "cap": 64, "config": cfg["midpoint"]}, indent=2))

    def decode(stage):
        rows = []
        for model in {id(m): m for m in models.values()}.values():
            generate(model, encode(evaluation[0]), 16)
        for i, record in enumerate(evaluation):
            names = list(models)
            names = names[i % len(names):]+names[:i % len(names)]
            ids = encode(record)
            for name in names:
                torch.cuda.synchronize(); begin = time.perf_counter()
                result = generate(models[name], ids, cfg["output_cap"])
                torch.cuda.synchronize(); seconds = time.perf_counter()-begin
                output = result.output_ids[0, ids.shape[1]:].tolist()
                rows.append({"stage": stage, "method": name, "benchmark": record["benchmark"], "problem_id": record["problem_id"], "seconds": seconds, "input_tokens": ids.shape[1], "output_tokens": len(output), "tokens": output, "acceptance_lengths": result.acceptance_lengths, "completion": tokenizer.decode(output, skip_special_tokens=True), "capped": len(output) >= cfg["output_cap"]})
        baseline = {r["problem_id"]: r for r in rows if r["method"] == "native"}
        for row in rows:
            if row["method"] == "duplicate" and any(row[k] != baseline[row["problem_id"]][k] for k in ["tokens", "acceptance_lengths"]):
                raise RuntimeError("Native duplicate output or acceptance mismatch")
        (args.output/f"decode-{stage}.json").write_text(json.dumps(rows, indent=2))
        summary = {}
        for name in models:
            arm = [r for r in rows if r["method"] == name]
            summary[name] = {"tps": sum(r["output_tokens"] for r in arm)/sum(r["seconds"] for r in arm), "progress": sum(sum(r["acceptance_lengths"]) for r in arm)/sum(len(r["acceptance_lengths"]) for r in arm), "exact_native": sum(r["tokens"] == baseline[r["problem_id"]]["tokens"] for r in arm), "requests": len(arm), "capped": sum(r["capped"] for r in arm)}
        for arm in summary.values():
            arm["native_ratio"] = arm["tps"]/summary["native"]["tps"]
        log("decode", stage=stage, summary=summary)
        return summary

    before = decode("before")
    cast_parameters(student, torch.float32).requires_grad_(cfg["train_mode"] == "joint")
    student.fc.requires_grad_(True)
    params = [p for p in student.parameters() if p.requires_grad]
    groups = [{"params": list(student.fc.parameters()), "lr": cfg["learning_rate"]*cfg.get("interface_lr_multiplier", 1.)}]
    rest = [p for name, p in student.named_parameters() if p.requires_grad and not name.startswith("fc.")]
    if rest:
        groups.append({"params": rest, "lr": cfg["learning_rate"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"], foreach=False)
    counts = {"native_parameters": sum(p.numel() for p in native.parameters()), "student_parameters": sum(p.numel() for p in student.parameters()), "trainable_parameters": sum(p.numel() for p in params)}
    log("optimizer", **counts)
    projection_start = student.fc.weight.detach().clone()
    draft_start = student.layers[0].mlp.down_proj.weight.detach().clone()

    def objective(records, anchors):
        view = block_batch(records, anchors, cfg["taps"], cfg["block_size"])
        prefix = cfg.get("conditioning_prefix", 0)
        tokens = conditioned_noise(view, native.mask_token_id, prefix).cuda()
        kwargs = dict(position_ids=view["positions"].cuda(), use_cache=False, is_causal=False)
        if len(records) > 1:
            kwargs["attention_mask"] = view["attention_mask"].cuda()
        with torch.no_grad():
            noise = target.model.embed_tokens(tokens)
            teacher = target.lm_head(view["teacher_hidden"].cuda())
            teacher_kind = cfg.get("teacher", "target")
            native_logits = None
            if teacher_kind in {"native", "blend"}:
                native_view = block_batch(records, anchors, TAPS, cfg["block_size"])
                native_hidden = native(target_hidden=native_view["context"].cuda(), noise_embedding=noise, **kwargs)
                native_logits = target.lm_head(native_hidden[:, 1:])
            elif teacher_kind != "target":
                raise ValueError("Invalid distillation teacher")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if midpoint is not None:
                midpoint.capture = True
            try:
                hidden = student(target_hidden=view["context"].cuda(), noise_embedding=noise, **kwargs)
            finally:
                if midpoint is not None:
                    midpoint.capture = False
            logits = target.lm_head(hidden[:, 1:])
        labels = view["labels"].cuda()[:, prefix:]
        logits, teacher = logits[:, prefix:], teacher[:, prefix:]
        target_loss, values = prediction_loss(logits, teacher, labels, cfg["loss"], cfg["gamma"], cfg["temperature"])
        loss = target_loss
        if native_logits is not None:
            native_loss, native_values = prediction_loss(logits, native_logits[:, prefix:], labels, "kl", cfg["gamma"], cfg["temperature"])
            values["native_kl"] = native_values["kl"]
            values["native_token_agreement"] = native_values["token_agreement"]
            loss = native_loss if teacher_kind == "native" else .5*(target_loss+native_loss)
        if midpoint is not None:
            middle = midpoint.take_logits()
            auxiliary, auxiliary_values = prediction_loss(middle, teacher[:, :midpoint.prefix], labels[:, :midpoint.prefix], cfg["midpoint"].get("auxiliary_loss", "ce"), cfg["gamma"], cfg["temperature"])
            loss = loss+cfg["midpoint"].get("auxiliary_weight", 1.)*auxiliary
            values["auxiliary_loss"] = float(auxiliary.detach())
            values["auxiliary_agreement"] = auxiliary_values["token_agreement"]
        return loss, values

    @torch.no_grad()
    def validate(step):
        metrics = []
        for path in datasets["validation"]:
            record = load_record(path)
            anchor = (record["anchor_min"]+record["anchor_max"])//2
            loss, values = objective([record], [anchor])
            metrics.append({"loss": float(loss), **values})
        summary = {key: sum(r[key] for r in metrics)/len(metrics) for key in metrics[0]}
        log("validation", step=step, **summary)
        return {"step": step, **summary}

    best_validation = None
    def retain_best(validation):
        nonlocal best_validation
        if not cfg.get("save_best"):
            return
        if best_validation is not None and validation["loss"] >= best_validation["loss"]:
            return
        best_path = scratch/"best-validation.pt"
        temporary = scratch/"best-validation.tmp"
        # Store deployable weights without changing live precision or buffers.
        state = {name: value.detach().to(device="cpu", dtype=torch.bfloat16) if value.is_floating_point() else value.detach().cpu() for name, value in student.state_dict().items()}
        torch.save({"config": cfg, "validation": validation, "state_dict": state}, temporary)
        temporary.replace(best_path)
        best_validation = {**validation, "path": str(best_path), "sha256": sha(best_path)}
        (args.output/"best-validation.json").write_text(json.dumps(best_validation, indent=2))

    validations = [validate(0)]
    retain_best(validations[-1])
    visited_records = set()
    with (args.output/"training.jsonl").open("w") as stream:
        for step in range(1, cfg["steps"]+1):
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.
            for micro in range(cfg["accumulation"]):
                start = ((step-1)*cfg["accumulation"]+micro)*batch_size
                records = [load_record(datasets["train"][(start+i) % len(datasets["train"])]) for i in range(batch_size)]
                visited_records.update(record["question_sha"] for record in records)
                anchors = [random.randint(record["anchor_min"], record["anchor_max"]) for record in records]
                loss, values = objective(records, anchors)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                (loss/cfg["accumulation"]).backward()
                loss_total += float(loss.detach())/cfg["accumulation"]
            norm = torch.nn.utils.clip_grad_norm_(params, cfg["gradient_clip"], error_if_nonfinite=True)
            if step == 1:
                if any(p.grad is not None for p in target.parameters()):
                    raise RuntimeError("Frozen target received gradients")
                if student.fc.weight.grad is None or not student.fc.weight.grad.abs().max() > 0:
                    raise RuntimeError("No interface gradient")
                if cfg["train_mode"] == "joint" and student.layers[0].mlp.down_proj.weight.grad is None:
                    raise RuntimeError("No joint draft gradient")
            optimizer.step()
            stream.write(json.dumps({"step": step, "loss": loss_total, "gradient_norm": float(norm), **values})+"\n"); stream.flush()
            if step % cfg["validate_every"] == 0 or step == cfg["steps"]:
                validations.append(validate(step))
                retain_best(validations[-1])
    counts["distinct_records_consumed"] = len(visited_records)
    counts["supervised_blocks"] = cfg["steps"]*cfg["accumulation"]*batch_size
    counts["supervised_token_positions"] = counts["supervised_blocks"]*(cfg["block_size"]-1-cfg.get("conditioning_prefix", 0))
    if cfg.get("require_all_records") and len(visited_records) != cfg["train_records"]:
        raise RuntimeError("Not all requested training records were consumed")
    changes = {"interface_l2": float((student.fc.weight.detach()-projection_start).norm()), "draft_l2": float((student.layers[0].mlp.down_proj.weight.detach()-draft_start).norm())}
    if changes["interface_l2"] == 0 or (cfg["train_mode"] == "joint" and changes["draft_l2"] == 0):
        raise RuntimeError("Requested trainable component did not update")
    if cfg["train_mode"] == "interface" and changes["draft_l2"] != 0:
        raise RuntimeError("Interface-only control changed draft weights")
    del optimizer, params, projection_start, draft_start
    student.zero_grad(set_to_none=True)
    cast_parameters(student, torch.bfloat16).eval().requires_grad_(False)
    torch.cuda.empty_cache()
    checkpoint = scratch/"student.pt"
    torch.save({"config": cfg, "state_dict": student.state_dict()}, checkpoint)
    (args.output/"checkpoint.json").write_text(json.dumps({"path": str(checkpoint), "sha256": sha(checkpoint)}, indent=2))
    after = decode("after")
    before_rows = json.loads((args.output/"decode-before.json").read_text())
    after_rows = json.loads((args.output/"decode-after.json").read_text())
    baseline_before = {r["problem_id"]: r for r in before_rows if r["method"] == "native"}
    for row in after_rows:
        if row["method"] == "native" and any(row[k] != baseline_before[row["problem_id"]][k] for k in ["tokens", "acceptance_lengths"]):
            raise RuntimeError("Untouched native control changed across training")
    restored = compact_student(native, cfg["taps"], cfg["layers"]).eval().requires_grad_(False)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    if cfg.get("midpoint"):
        restored_midpoint = MidpointConditioning(restored, target, cfg["midpoint"])
    probe_cap = min(cfg["output_cap"], 64)
    live = generate(student, encode(evaluation[0]), probe_cap)
    reloaded = generate(restored, encode(evaluation[0]), probe_cap)
    if not torch.equal(live.output_ids, reloaded.output_ids) or live.acceptance_lengths != reloaded.acceptance_lengths:
        raise RuntimeError("Saved checkpoint failed decode reproduction")
    (args.output/"reload-gate.json").write_text(json.dumps({"status": "pass", "cap": probe_cap, "output_tokens": live.num_output_tokens, "checkpoint_sha256": sha(checkpoint)}, indent=2))
    result = {"status": "pass", "before": before, "after": after, "validation": validations, "best_validation": best_validation, "evaluated_checkpoint": "final", "updates": changes, "counts": counts, "elapsed_seconds": time.perf_counter()-started, "cuda_max_allocated": torch.cuda.max_memory_allocated()}
    (args.output/"result.json").write_text(json.dumps(result, indent=2))
    log("complete", summary=result)


if __name__ == "__main__":
    main()
