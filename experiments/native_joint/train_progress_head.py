"""Fit a small native draft head to actual prefix-valid verification errors."""
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

from core import ProgressHead, cast_parameters, progress_loss, token_ids, stop_aware_prefix_mask
from radical_decode import decode_variant
from run_lane import TARGET, DRAFT, COMMIT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("One GPU per fit")
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
    eos = target.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    index = json.loads(Path(cfg["progress_index"]).read_text())
    if index["status"] != "pass" or index["target"] != TARGET or index["draft"] != DRAFT:
        raise RuntimeError("Verification cache provenance mismatch")
    datasets, questions, question_order = {}, {}, {}
    post_eos_removed = 0
    for split in ["train", "validation"]:
        chunks = []
        for entry in index["entries"]:
            if entry["split"] != split:
                continue
            if sha(entry["path"]) != entry["sha256"]:
                raise RuntimeError("Verification cache hash mismatch")
            chunk = torch.load(entry["path"], map_location="cpu", weights_only=True)
            before_valid = int(chunk["valid"].sum())
            chunk["valid"] = stop_aware_prefix_mask(chunk["valid"], chunk["native_tokens"], eos)
            immediate_stops = {r["question_sha"] for r in entry["records"] if r["output_tokens"] == 1}
            for i, question in enumerate(chunk["question_shas"]):
                if question in immediate_stops:
                    chunk["valid"][i] = False
            post_eos_removed += before_valid-int(chunk["valid"].sum())
            chunks.append(chunk)
        if not chunks:
            raise RuntimeError("Missing split")
        datasets[split] = {key: torch.cat([c[key] for c in chunks]) for key in ["hidden", "labels", "valid", "native_tokens"]}
        question_order[split] = [q for c in chunks for q in c["question_shas"]]
        questions[split] = set(question_order[split])
    if questions["train"] & questions["validation"]:
        raise RuntimeError("Training/validation question overlap")
    head = ProgressHead(native.config.hidden_size, cfg["rank"], cfg.get("positions")).cuda().bfloat16().eval()
    records = [r for benchmark in cfg["benchmarks"] for r in [r for r in json.loads(Path(cfg["eval_manifest"]).read_text())["records"] if r["benchmark"] == benchmark][:cfg["requests_per_benchmark"]]]
    def encode(record):
        prompt = record.get("prompt") or record["turns"][0]
        return torch.tensor(token_ids(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=False)), device="cuda").unsqueeze(0)
    def generate(ids, cap, correction=None):
        if correction is None:
            return official.dflash_generate(native, target, ids, cap, eos, 0., block_size=16, return_stats=True)
        return decode_variant(official, native, target, ids, cap, eos, {"kind": "head"}, head=correction)
    native_gate = {}
    for record in records:
        ids = encode(record)
        original, zero = generate(ids, 64), generate(ids, 64, head)
        if not torch.equal(original.output_ids, zero.output_ids) or original.acceptance_lengths != zero.acceptance_lengths:
            raise RuntimeError("Zero correction failed native-equivalence gate")
        native_gate[record["problem_id"]] = original.output_ids[0, ids.shape[1]:].tolist()
    cast_parameters(head, torch.float32).requires_grad_(True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["learning_rate"], weight_decay=cfg.get("weight_decay", 0.))
    def objective(split, indices):
        batch = {key: value[indices].cuda() for key, value in datasets[split].items()}
        if cfg.get("positions") is not None:
            position_mask = torch.zeros_like(batch["valid"])
            position_mask[:, cfg["positions"]] = True
            batch["valid"] = batch["valid"] & position_mask
        hidden = batch["hidden"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            corrected = head(hidden)
            logits = target.lm_head(corrected)
        native_logits = None
        if cfg.get("loss_kind") == "margin_kl":
            with torch.no_grad():
                native_logits = target.lm_head(hidden)
        loss, metrics = progress_loss(logits, batch["labels"], batch["valid"], batch["native_tokens"], cfg["error_weight"], kind=cfg.get("loss_kind", "ce"), native_logits=native_logits, margin=cfg.get("margin", .1), preserve_weight=cfg.get("preserve_weight", 1.), soft_temperature=cfg.get("soft_temperature", .1))
        delta = ((corrected.float()-hidden.float()).square().sum(-1)/(hidden.float().square().sum(-1)+1e-6)).mean()
        return loss+cfg.get("regularization", 0.)*delta, {**metrics, "relative_delta": float(delta.detach())}
    @torch.no_grad()
    def validate(step):
        values, losses = [], []
        for start in range(0, len(datasets["validation"]["labels"]), cfg["batch_size"]):
            indices = torch.arange(start, min(start+cfg["batch_size"], len(datasets["validation"]["labels"])))
            loss, metrics = objective("validation", indices)
            values.append(metrics); losses.append((float(loss), len(indices)))
        sums = {key: sum(m[key] for m in values) for key in ["valid_positions", "correct_valid", "first_errors", "first_errors_repaired", "preserved", "originally_correct"]}
        result = {"step": step, "loss": sum(loss*n for loss, n in losses)/sum(n for _, n in losses), **sums}
        print(json.dumps({"event": "validation", "elapsed": time.perf_counter()-started, **result}), flush=True)
        return result
    validation = [validate(0)]
    rng = torch.Generator().manual_seed(cfg["seed"])
    question_blocks = {}
    for i, question in enumerate(question_order["train"]):
        question_blocks.setdefault(question, []).append(i)
    question_keys = sorted(question_blocks)
    question_ordering = torch.randperm(len(question_keys), generator=rng).tolist() if cfg.get("sampling") == "question_uniform" else []
    visited = set()
    with (args.output/"training.jsonl").open("w") as stream:
        for step in range(1, cfg["steps"]+1):
            if cfg.get("sampling") == "question_uniform":
                choices = []
                for j in range(cfg["batch_size"]):
                    key = question_keys[question_ordering[((step-1)*cfg["batch_size"]+j)%len(question_keys)]]
                    blocks = question_blocks[key]
                    choices.append(blocks[int(torch.randint(len(blocks), (), generator=rng))])
                indices = torch.tensor(choices)
            else:
                indices = torch.randint(len(datasets["train"]["labels"]), (cfg["batch_size"],), generator=rng)
            visited.update(indices.tolist())
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = objective("train", indices)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            if step == 1 and (any(p.grad is not None for p in target.parameters()) or any(p.grad is not None for p in native.parameters())):
                raise RuntimeError("Frozen native or target received gradients")
            optimizer.step()
            stream.write(json.dumps({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm), **metrics})+"\n"); stream.flush()
            if step%cfg["validate_every"] == 0 or step == cfg["steps"]:
                validation.append(validate(step))
    if not head.up.weight.abs().max() > 0:
        raise RuntimeError("Correction did not train")
    del optimizer
    head.zero_grad(set_to_none=True)
    cast_parameters(head, torch.bfloat16).eval().requires_grad_(False)
    scratch = Path(os.environ["NATIVE_SCRATCH"])/os.environ["SLURM_JOB_ID"]/cfg.get("scratch_tag", args.output.name)
    scratch.mkdir(parents=True, exist_ok=False)
    checkpoint = scratch/"progress-head.pt"
    torch.save({"config": cfg, "state_dict": head.state_dict()}, checkpoint)
    restored = ProgressHead(native.config.hidden_size, cfg["rank"], cfg.get("positions")).cuda().bfloat16().eval().requires_grad_(False)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    probe = encode(records[0])
    live, reloaded = generate(probe, 64, head), generate(probe, 64, restored)
    if not torch.equal(live.output_ids, reloaded.output_ids) or live.acceptance_lengths != reloaded.acceptance_lengths:
        raise RuntimeError("Saved head failed decode reproduction")
    del restored
    rows = []
    for repeat in range(cfg.get("repeats", 1)):
        for i, record in enumerate(records):
            ids = encode(record)
            for name in (["native", "candidate"] if (i+repeat)%2 == 0 else ["candidate", "native"]):
                torch.cuda.synchronize(); begin = time.perf_counter()
                generated = generate(ids, cfg["output_cap"], None if name == "native" else head)
                torch.cuda.synchronize(); seconds = time.perf_counter()-begin
                tokens = generated.output_ids[0, ids.shape[1]:].tolist()
                row = {"method": name, "repeat": repeat, "benchmark": record["benchmark"], "problem_id": record["problem_id"], "tokens": tokens, "output_tokens": len(tokens), "seconds": seconds, "acceptance_lengths": generated.acceptance_lengths, "trace": getattr(generated, "trace", None)}
                rows.append(row)
                with (args.output/"evaluation.jsonl").open("a") as stream:
                    stream.write(json.dumps(row)+"\n")
    baseline = {(r["repeat"], r["problem_id"]): r for r in rows if r["method"] == "native"}
    initial = {(r["method"], r["problem_id"]): r for r in rows if r["repeat"] == 0}
    for row in rows:
        if any(row[k] != initial[row["method"], row["problem_id"]][k] for k in ["tokens", "acceptance_lengths"]):
            raise RuntimeError("Repeated evaluation changed output or acceptance")
        reference = native_gate[row["problem_id"]]
        if row["method"] == "native" and row["tokens"][:len(reference)] != reference:
            raise RuntimeError("Native output changed across correction training")
    summary = {}
    for name in ["native", "candidate"]:
        group = [r for r in rows if r["method"] == name]
        summary[name] = {"tps": sum(r["output_tokens"] for r in group)/sum(r["seconds"] for r in group), "exact_native": sum(r["tokens"] == baseline[r["repeat"], r["problem_id"]]["tokens"] for r in group), "requests": len(records), "timed_generations": len(group)}
    result = {"status": "pass", "config": cfg, "target": TARGET, "draft": DRAFT, "source_commit": COMMIT, "source_file_sha256": sha(Path(source)/"dflash/model.py"), "gpu": torch.cuda.get_device_name(), "job": os.environ["SLURM_JOB_ID"], "eval_manifest_sha256": sha(cfg["eval_manifest"]), "progress_index_sha256": sha(cfg["progress_index"]), "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)}, "zero_gate": "pass", "reload_gate": "pass", "validation": validation, "summary": summary, "native_ratio": summary["candidate"]["tps"]/summary["native"]["tps"], "counts": {"parameters": sum(p.numel() for p in head.parameters()), "train_questions": len(questions["train"]), "distinct_questions_consumed": len({question_order["train"][i] for i in visited}), "val_questions": len(questions["validation"]), "train_blocks": len(datasets["train"]["labels"]), "distinct_blocks_consumed": len(visited), "sampled_blocks": cfg["steps"]*cfg["batch_size"]}, "elapsed_seconds": time.perf_counter()-started}
    result["post_eos_positions_removed"] = post_eos_removed
    result["host"] = os.environ.get("SLURMD_NODENAME", os.environ.get("SLURM_NODELIST"))
    (args.output/"head-result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
