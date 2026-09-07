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
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))
    pair = "llama" if rank < 2 else "cross"
    shard = rank % 2
    root = Path(os.environ["ZIP_FAMILY_OUTPUT"]) / pair
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
    manifest = Path("configs/train_math_4096.json")
    rows = json.loads(manifest.read_text())["records"][:512]
    assert len(rows) == 512
    collected = []
    start = time.perf_counter()
    for i in range(shard, len(rows), 2):
        row = rows[i]
        # Both encoders see IDENTICAL plain content. Shared token end offsets
        # pair hidden states after the same text prefix, never span-overlap.
        text = row["problem"] + "\n\n" + row["solution"]
        enc = [tok(text, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=1024) for tok in (tt, st)]
        ends = [{b: j for j, (a, b) in enumerate(e["offset_mapping"]) if b > a} for e in enc]
        common = sorted(set(ends[0]) & set(ends[1]))
        assert common, f"No common text boundary: {i}"
        gen = torch.Generator().manual_seed(42 + i)
        selected = torch.randperm(len(common), generator=gen)[:max(1, math.ceil(len(common) / 4))].sort().values.tolist()
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
        collected.append({"row": i, "x": feats[0], "y": feats[1], "boundaries": boundaries})
        if len(collected) % 32 == 0:
            print(json.dumps({"pair": pair, "rank": rank, "captured": len(collected), "seconds": time.perf_counter()-start}), flush=True)
    torch.save(collected, root / f"features-{shard}.pt")
    del source, target, collected
    gc.collect(); torch.cuda.empty_cache()
    dist.barrier()
    if shard == 0:
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        records = sum([torch.load(root / f"features-{j}.pt", weights_only=True) for j in range(2)], [])
        records.sort(key=lambda r: r["row"])
        assert [r["row"] for r in records] == list(range(512))
        x = torch.cat([r["x"] for r in records]); y = torch.cat([r["y"] for r in records])
        w = torch.cat([torch.full((len(r["x"]),), 1/len(r["x"])) for r in records])
        del records
        m = Context(fusion, norm, d8=di, d4=do, layers=5).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=0, fused=True)
        batch = 2048; steps = math.ceil(len(x)/batch)*3; warm = max(1, int(.05*steps)); step = 0
        history = []; started = time.perf_counter()
        for epoch in range(3):
            order = torch.randperm(len(x), generator=torch.Generator().manual_seed(42+epoch))
            total = 0.; epoch_start = time.perf_counter()
            for inds in order.split(batch):
                rate = (step+1)/warm if step<warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,steps-warm)))
                opt.param_groups[0]["lr"] = .001*rate
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = m.loss(x[inds].cuda(), y[inds].cuda())
                weighted = (losses*w[inds].cuda()).sum()
                loss = weighted*(len(x)/512)/batch
                assert torch.isfinite(loss)
                loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True); opt.step()
                total += weighted.item(); step += 1
            event = {"epoch": epoch+1, "loss": total/512, "seconds": time.perf_counter()-epoch_start}
            history.append(event); print(json.dumps({"pair": pair, **event}), flush=True)
        folded = m.folded().to(torch.bfloat16)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            sample = x[:256].cuda(); a, _ = m(sample)
            b = m.frozen_norm(torch.nn.functional.linear(sample, folded))
            fold_error = ((a.float()-b.float()).square().sum(-1)/(a.float().square().sum(-1)+1e-6)).mean().item()
        assert fold_error < .001, fold_error
        torch.save({"model": {k:v.cpu() for k,v in m.state_dict().items()}, "config": cfg}, root/"maps.pt")
        torch.save({"relay": {"projection.weight": folded.cpu()}, "target_layer_ids": target_taps, "source_layer_ids": source_taps, "relay_architecture": "raw_linear", "proposer_family": "dflash", "feature_objective": "zip_layer_plus_context", "steps": step}, root/"relay.pt")
        summary = {"pair": pair, "records":512, "positions":len(x), "history":history, "train_seconds":time.perf_counter()-started, "fold_relative_mse":fold_error, "alignment":"identical plain text, exact shared token-end boundaries", "max_input_tokens":1024, "sample_fraction":.25, "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(), "job_id":os.environ["SLURM_JOB_ID"], "config":cfg, "scope":"ZIP objective and optimizer pilot; existing solution text, not generated rollout reproduction"}
        (root/"summary.json").write_text(json.dumps(summary, indent=2))
    dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__":
    main()
