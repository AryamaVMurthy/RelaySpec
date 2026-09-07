"""ZIP objective pilots; existing calibration text, explicitly not rollout reproduction."""
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from relayspec.dflash import import_official_dflash
from relayspec.relay import extract_hidden_taps

# Import the unmodified archived mathematical objective.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external/transfer-reproduction/transfer/src"))
from mapper import Context


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=16384)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    pair = "llama" if rank < 2 else "cross"
    shard = rank % 2
    root = Path(os.environ["FAMILY_SCALE_CACHE"]) / pair
    root.mkdir(parents=True, exist_ok=True)
    filename = "train_dflash_llama32_3b_x2_4gpu.yaml" if pair == "llama" else "train_x1_qwen_proposer_to_llama31_8b_aligned_4gpu.yaml"
    cfg = yaml.safe_load((Path("configs/protocol_next") / filename).read_text())
    cache = os.environ["TRANSFORMERS_CACHE"]
    source_spec = cfg["source_trunk"]["model"]
    target_spec = cfg["target"]
    def tokenizer(spec):
        return AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], cache_dir=cache, local_files_only=True)
    st, tt = tokenizer(source_spec), tokenizer(target_spec)
    def model(spec):
        return AutoModelForCausalLM.from_pretrained(spec["id"], revision=spec["revision"], cache_dir=cache, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    source, target = model(source_spec), model(target_spec)
    dc, _ = import_official_dflash(os.environ["DFLASH_SOURCE"], cfg["proposer"]["source_commit"])
    draft = dc.from_pretrained(cfg["proposer"]["id"], revision=cfg["proposer"]["revision"], cache_dir=cache, local_files_only=True, dtype=torch.bfloat16).cuda().eval().requires_grad_(False)
    assert draft.hidden_norm.variance_epsilon == 1e-6
    source_taps = tuple(draft.target_layer_ids)
    target_taps = tuple(cfg["relay_training"]["target_layer_ids"])
    di, do = target.config.hidden_size, source.config.hidden_size
    fusion, norm = draft.fc.weight.detach().float().cpu(), draft.hidden_norm.weight.detach().float().cpu()
    del draft
    gc.collect(); torch.cuda.empty_cache()
    data = Path(os.environ["FAMILY_SCALE_DATA"])
    gate = json.loads((data / "manifest-gate.json").read_text())
    def read(name):
        raw = (data/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == gate["files"][name]["sha256"]
        return json.loads(raw)["records"]
    train = read("train-16384.json"); validation = read("validation.json")[:256]
    assert read("train-8192.json") == train[:8192]
    assert len({r["problem"] for r in train}) == 16384
    assert not ({r["problem"] for r in train} & {r["problem"] for r in validation})
    rows = [("train", i, r) for i,r in enumerate(train[:args.limit])]
    rows += [("validation", i, r) for i,r in enumerate(validation)]
    metadata = {"config":cfg, "fusion":fusion, "norm":norm, "di":di, "do":do,
        "source_taps":source_taps, "target_taps":target_taps,
        "train_sha256":gate["files"]["train-16384.json"]["sha256"],
        "validation_sha256":gate["files"]["validation.json"]["sha256"],
        "positions_per_record":32,"max_input_tokens":1024}
    if shard == 0:
        torch.save(metadata, root/"metadata.pt")
    captured_count = 0
    start = time.perf_counter()
    for split, i, row in rows[shard::2]:
        dest = root / split / f"{i:05d}.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        row_hash = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        if dest.exists():
            old = torch.load(dest, weights_only=True, mmap=True)
            assert old["row_sha256"] == row_hash and old["x"].shape[0] <= 32
            continue
        # Both encoders see IDENTICAL plain content. Shared token end offsets
        # pair hidden states after the same text prefix, never span-overlap.
        text = row["problem"] + "\n\n" + row["solution"]
        enc = [tok(text, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=1024) for tok in (tt, st)]
        ends = [{b: j for j, (a, b) in enumerate(e["offset_mapping"]) if b > a} for e in enc]
        common = sorted(set(ends[0]) & set(ends[1]))
        assert common, f"No common text boundary: {i}"
        gen = torch.Generator().manual_seed(42 + i)
        selected = torch.randperm(len(common), generator=gen)[:min(32, len(common))].sort().values.tolist()
        boundaries = [common[j] for j in selected]
        feats = []
        with torch.inference_mode():
            for m, e, index, taps in zip((target, source), enc, ends, (target_taps, source_taps)):
                ids = torch.tensor([e["input_ids"]], device="cuda")
                captured = {}
                def hook(k):
                    def capture(module, args, output):
                        captured[k] = output[0] if isinstance(output, tuple) else output
                    return capture
                handles = [m.model.layers[k].register_forward_hook(hook(k)) for k in taps]
                try:
                    out = m(ids, use_cache=False, logits_to_keep=1)
                finally:
                    for handle in handles:
                        handle.remove()
                h = torch.cat([captured[k] for k in taps], dim=-1)
                feats.append(h[0, [index[b] for b in boundaries]].cpu())
                del out, h, captured
        assert all(torch.isfinite(f).all() for f in feats)
        result = {"row": i, "split":split, "row_sha256":row_hash,
                  "x":feats[0], "y":feats[1], "boundaries":boundaries}
        temporary = dest.with_suffix(".tmp")
        torch.save(result, temporary); temporary.replace(dest)
        captured_count += 1
        if captured_count % 32 == 0:
            print(json.dumps({"pair":pair,"rank":rank,"captured":captured_count,
                "seconds":time.perf_counter()-start}),flush=True)
    (root/f"capture-{shard}-{args.limit}.json").write_text(json.dumps({
        "status":"complete", "train_records":args.limit,"validation_records":256,
        "captured_now":captured_count,"seconds":time.perf_counter()-start,
        "job_id":os.environ["SLURM_JOB_ID"]},indent=2))

if __name__ == "__main__":
    main()
