"""Capture native proposal states and prefix-valid target verification labels."""
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

from run_lane import TARGET, DRAFT, COMMIT, sha
from core import token_ids, TAPS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("One GPU per collection lane")
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
    eos = target.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    source_path = cfg.get("prompt_manifest") or cfg["prepared_index"]
    prepared = json.loads(Path(source_path).read_text())
    if prepared["status"] != "pass" or (not cfg.get("prompt_manifest") and prepared["target"] != TARGET):
        raise RuntimeError("Prompt provenance mismatch")
    tokenizer = None
    if cfg.get("prompt_manifest"):
        if sha(source_path) != cfg["prompt_manifest_sha256"]:
            raise RuntimeError("Prompt manifest changed")
        tokenizer = AutoTokenizer.from_pretrained(TARGET["id"], revision=TARGET["revision"], cache_dir=os.environ["HF_HUB_CACHE"], local_files_only=True)
    scratch = Path(os.environ["NATIVE_SCRATCH"])/os.environ["SLURM_JOB_ID"]/args.output.name
    scratch.mkdir(parents=True, exist_ok=False)
    state = {}
    def draft_hook(module, inputs, output):
        state["hidden"] = output[0, 1:].detach().cpu()
    def target_hook(module, inputs, kwargs, output):
        if "hidden" not in state:
            return
        proposal = (inputs[0] if inputs else kwargs["input_ids"])[0, 1:]
        labels = output.logits[0, :-1].argmax(-1)
        if len(proposal) != 15 or len(labels) != 15:
            raise RuntimeError("Unexpected native verification block shape")
        matches = proposal == labels
        valid = torch.cat([torch.ones(1, device=labels.device, dtype=torch.bool), matches.cumprod(0)[:-1].bool()])
        state["blocks"].append({"hidden": state.pop("hidden"), "labels": labels.cpu(), "valid": valid.cpu(), "native_tokens": proposal.cpu(), "question_sha": state["question_sha"]})
    handles = [native.register_forward_hook(draft_hook), target.register_forward_hook(target_hook, with_kwargs=True)]
    outputs, used = [], set()
    sequence_entries, sequence_excluded = [], []
    try:
        for split, count in [("train", cfg["train_records"]), ("validation", cfg["val_records"])]:
            pool = ([e for e in prepared["records"] if e["split"] == split and e["domain"] == cfg["domain"]] if tokenizer is not None else sorted([e for e in prepared["entries"] if e["split"] == split], key=lambda e: e["index"]))
            selected = [entry for i, entry in enumerate(pool) if i%cfg["shards"] == cfg["shard"]][:count]
            if len(selected) != count:
                raise RuntimeError("Not enough disjoint source prompts")
            state["blocks"] = []
            records = []
            for entry in selected:
                if entry["question_sha"] in used:
                    raise RuntimeError("Duplicate prompt")
                used.add(entry["question_sha"])
                if tokenizer is not None:
                    ids = torch.tensor(token_ids(tokenizer.apply_chat_template([{"role": "user", "content": entry["problem"]}], tokenize=True, add_generation_prompt=True, enable_thinking=False)), device="cuda").unsqueeze(0)
                else:
                    if sha(entry["path"]) != entry["sha256"]:
                        raise RuntimeError("Source cache hash mismatch")
                    payload = torch.load(entry["path"], map_location="cpu", weights_only=True)
                    ids = payload["input_ids"][:payload["anchor_min"]].unsqueeze(0).cuda()
                state["question_sha"] = entry["question_sha"]
                before = len(state["blocks"])
                generated = official.dflash_generate(native, target, ids, cfg["output_cap"], eos, 0., block_size=16, return_stats=True)
                if len(state["blocks"])-before != len(generated.acceptance_lengths):
                    raise RuntimeError("Proposal/verification capture counts differ")
                for block, accepted in zip(state["blocks"][before:], generated.acceptance_lengths):
                    if int(block["valid"].sum()) != min(accepted, 15):
                        raise RuntimeError("Prefix labels extend beyond first rejection")
                records.append({"question_sha": entry["question_sha"], "input_tokens": ids.shape[1], "output_tokens": generated.num_output_tokens, "blocks": len(state["blocks"])-before})
                if cfg.get("prepare_sequences"):
                    sequence = generated.output_ids[0, :cfg["sequence_length"]].detach()
                    lo, hi = ids.shape[1], len(sequence)-16
                    if lo > hi:
                        sequence_excluded.append({"split": split, "question_sha": entry["question_sha"], "reason": "No full answer block within sequence cap"})
                    else:
                        with torch.no_grad():
                            features = target.model(input_ids=sequence.unsqueeze(0), use_cache=False, output_hidden_states=True)
                            taps = torch.stack([features.hidden_states[i+1][0] for i in TAPS]).cpu()
                            final_hidden = features.last_hidden_state[0].cpu()
                            if not sequence_entries:
                                altered = sequence.clone()
                                altered[lo+1:] = 0
                                check = target.model(input_ids=altered.unsqueeze(0), use_cache=False, output_hidden_states=True)
                                if not torch.equal(check.last_hidden_state[0, :lo+1].cpu(), final_hidden[:lo+1]) or not all(torch.equal(check.hidden_states[t+1][0, :lo].cpu(), taps[j, :lo]) for j, t in enumerate(TAPS)):
                                    raise RuntimeError("Future rollout tokens affected prefix features")
                                (args.output/"sequence-causality-gate.json").write_text(json.dumps({"status": "pass", "anchor": lo, "tokens": len(sequence)}))
                                del check
                        sequence_path = scratch/f"sequence-{split}-{len(sequence_entries):05}.pt"
                        payload = {"input_ids": sequence.cpu(), "taps": taps, "final_hidden": final_hidden, "anchor_min": lo, "anchor_max": hi, "question_sha": entry["question_sha"]}
                        torch.save(payload, sequence_path)
                        sequence_entries.append({"split": split, "index": len(sequence_entries), "domain": cfg.get("domain", "math"), "question_sha": entry["question_sha"], "path": str(sequence_path), "sha256": sha(sequence_path), "tokens": len(sequence), "input_ids": sequence.tolist(), "anchor_min": lo, "anchor_max": hi})
                        (args.output/"sequence-cache.json").write_text(json.dumps(sequence_entries, indent=2))
                        del features, payload, taps, final_hidden
            blocks = state["blocks"]
            path = scratch/f"{split}.pt"
            torch.save({"hidden": torch.stack([b["hidden"] for b in blocks]), "labels": torch.stack([b["labels"] for b in blocks]), "valid": torch.stack([b["valid"] for b in blocks]), "native_tokens": torch.stack([b["native_tokens"] for b in blocks]), "question_shas": [b["question_sha"] for b in blocks]}, path)
            outputs.append({"split": split, "path": str(path), "sha256": sha(path), "records": records, "blocks": len(blocks), "supervised_positions": sum(int(b["valid"].sum()) for b in blocks)})
            print(json.dumps({"event": "captured", "split": split, "blocks": len(blocks), "elapsed": time.perf_counter()-started}), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    result = {"status": "pass", "config": cfg, "target": TARGET, "draft": DRAFT, "source_commit": COMMIT, "prompt_source_sha256": sha(source_path), "outputs": outputs, "elapsed_seconds": time.perf_counter()-started, "scope": "Native greedy rollouts from disjoint raw training/validation prompts. Only actual accepted-prefix labels and the first rejection are supervised; hypothetical post-rejection target labels are masked. Generated features past the requested cap can occur in the last verification block and are retained only when prefix-valid."}
    (args.output/"progress-data.json").write_text(json.dumps(result, indent=2))
    if cfg.get("prepare_sequences"):
        (args.output/"sequence-result.json").write_text(json.dumps({"status": "pass", "config": cfg, "target": TARGET, "draft": DRAFT, "source_commit": COMMIT, "prompt_source_sha256": sha(source_path), "entries": sequence_entries, "excluded": sequence_excluded, "scope": "Frozen native greedy rollout sequences, followed by a fresh causal target feature pass. These are teacher-forced features on native-generated text, not online acceptance labels; only prefix features are exposed to a training block."}, indent=2))


if __name__ == "__main__":
    main()
