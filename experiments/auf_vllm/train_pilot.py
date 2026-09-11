"""Measured Q8 pilot: frozen DFlash, original targets, matched ZIP/CE/AUF maps."""
import argparse
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from .blocks import make_block, collate_blocks, conditioned_forward
from .interfaces import LayerContextMapper
from .losses import token_loss
from .pilot_data import write, sha


def tensor(path, name):
    index = path/"model.safetensors.index.json"
    file = path/json.loads(index.read_text())["weight_map"][name] if index.exists() else path/"model.safetensors"
    with safe_open(file, framework="pt", device="cpu") as reader:
        return reader.get_tensor(name)


def load_models(work):
    from transformers import Qwen3Config
    sys.path.insert(0, os.environ["DFLASH_SOURCE"])
    cls = importlib.import_module("dflash.model").DFlashDraftModel
    source = work/"models/4b/draft"
    config = Qwen3Config.from_pretrained(source, local_files_only=True)
    config._attn_implementation = "sdpa"
    draft = cls(config)
    state = load_file(str(source/"model.safetensors"))
    draft.load_state_dict(state, strict=True, assign=True)
    draft = draft.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    embedding = tensor(work/"models/4b/target", "model.embed_tokens.weight").to("cuda", torch.bfloat16)
    assert not embedding.requires_grad
    return draft, embedding


def logits(draft, embedding, mapper, batch):
    context, _ = mapper(batch["context"])
    noise = F.embedding(batch["noise_ids"], embedding)
    hidden = conditioned_forward(draft, context, noise, batch["position_ids"], batch["attention_mask"])
    return F.linear(hidden, embedding)


def load_examples(root, split):
    rows = json.loads((root/f"{split}.json").read_text())
    examples = []
    for index, row in enumerate(rows):
        target = torch.load(root/f"features/8/{split}/{index:05d}.pt", weights_only=True)
        source = torch.load(root/f"features/4/{split}/{index:05d}.pt", weights_only=True)
        assert target["group_id"] == source["group_id"] == row["group_id"]
        examples.append((row, target["features"], source["features"]))
    return examples


def block_for(example, anchor, draft):
    row, features, _ = example
    return make_block(features, row["full_ids"], anchor, draft.block_size, draft.mask_token_id,
                      (draft.config.eos_token_id,))


def batch_gate(draft, embedding, mapper, examples):
    # Unequal prefix lengths exercise padding and RoPE indexing.
    blocks = [block_for(examples[i], len(examples[i][0]["prompt_token_ids"])+i*3, draft) for i in range(2)]
    batch = collate_blocks(blocks, "cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        joined = logits(draft, embedding, mapper, batch)
        separate = torch.cat([logits(draft, embedding, mapper, collate_blocks([block], "cuda")) for block in blocks])
        context, _ = mapper(batch["context"])
        noise = F.embedding(batch["noise_ids"], embedding)
        # Independent official forward with source-space features checks helper parity.
        source_features = torch.randn(1, 5, 12800, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(5+draft.block_size, device="cuda")[None]
        noise_one = noise[:1]
        direct = draft(position_ids=positions, noise_embedding=noise_one, target_hidden=source_features,
                       attention_mask=None, use_cache=False, is_causal=False)
        helper = conditioned_forward(draft, draft.hidden_norm(draft.fc(source_features)), noise_one, positions, None)
        torch.testing.assert_close(direct, helper, rtol=0, atol=0)
        perturbed = dict(batch)
        perturbed["context"] = batch["context"].clone()
        for i, block in enumerate(blocks):
            perturbed["context"][i, block.anchor:] = 100*torch.randn_like(perturbed["context"][i, block.anchor:])
        padded = logits(draft, embedding, mapper, perturbed)
        torch.testing.assert_close(joined, padded, rtol=0, atol=0)
    relative = ((joined.float()-separate.float()).square().sum()/separate.float().square().sum()).item()
    # Compare supported labels only; rounding near an argmax tie is reported separately.
    agreement = (joined.argmax(-1)[batch["valid"]] == separate.argmax(-1)[batch["valid"]]).float().mean().item()
    # FP32 isolates indexing/masking errors from BF16 shape-dependent GEMMs.
    draft.float()
    mapper.fusion = mapper.fusion.float()
    mapper.norm = mapper.norm.float()
    fp_batch = dict(batch)
    fp_batch["context"] = batch["context"].float()
    embedding_fp = embedding.float()
    with torch.no_grad():
        fp_joined = logits(draft, embedding_fp, mapper, fp_batch)
        singles = []
        for block in blocks:
            one = collate_blocks([block], "cuda")
            one["context"] = one["context"].float()
            singles.append(logits(draft, embedding_fp, mapper, one))
        fp_separate = torch.cat(singles)
        fp_relative = ((fp_joined-fp_separate).square().sum()/fp_separate.square().sum()).item()
        fp_agreement = (fp_joined.argmax(-1)[batch["valid"]] == fp_separate.argmax(-1)[batch["valid"]]).float().mean().item()
        joined_support = token_loss(joined,batch["labels"],batch["valid"],"auf").active
        separate_support = token_loss(separate,batch["labels"],batch["valid"],"auf").active
        support_differences = int((joined_support != separate_support).sum())
    report = {"bf16_relative_mse":relative,"bf16_argmax_agreement":agreement,
              "fp32_relative_mse":fp_relative,"fp32_argmax_agreement":fp_agreement,
              "bf16_support_differences":support_differences}
    print(json.dumps({"batch_numerics":report}),flush=True)
    assert fp_relative < 1e-10 and fp_agreement == 1.0, report
    # Diagnosed at job31250: FP32 relative L2=1.76e-6, while BF16 shape
    # rounding gives 1.34%; greedy labels and AUF support agree exactly.
    # Use 2% BF16 relative L2 only together with the strict structural checks.
    # Exact BF16 argmax invariance is not a valid structural requirement near
    # token ties. Fixed batching defines training support; report its shape
    # sensitivity separately. FP32 argmax and all masking/helper checks must pass.
    assert relative < 4e-4, report
    draft.bfloat16()
    mapper.fusion = mapper.fusion.bfloat16()
    mapper.norm = mapper.norm.bfloat16()
    del embedding_fp, fp_joined, fp_separate, singles
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = token_loss(logits(draft, embedding, mapper, batch), batch["labels"], batch["valid"], "auf")
    result.loss.backward()
    grads = [p.grad for p in mapper.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.float().square().sum().item() for g in grads) > 0
    assert all(p.grad is None and not p.requires_grad for p in draft.parameters())
    mapper.zero_grad(set_to_none=True)
    return {"passed": True, "batch_relative_mse": relative, "batch_argmax_agreement": agreement,
            "fp32_batch_relative_mse":fp_relative,"fp32_argmax_agreement":fp_agreement,
            "bf16_support_differences":support_differences,"bf16_relative_l2_tolerance":.02,
            "official_helper_exact": True, "padding_perturbation_exact": True,
            "active_tokens": int(result.active.sum()), "frozen_draft_gradients": "none"}


def evaluate(draft, embedding, mapper, examples):
    totals = {"ce":0., "auf":0., "feature":0., "prefix_tokens":0, "blocks":0}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for example in examples:
            row, x, y = example
            anchor = len(row["prompt_token_ids"])
            batch = collate_blocks([block_for(example, anchor, draft)], "cuda")
            scores = logits(draft, embedding, mapper, batch)
            for name in ["ce", "auf"]:
                loss = token_loss(scores, batch["labels"], batch["valid"], name)
                totals[name] += loss.loss.item()
            valid_correct = (~batch["valid"] | (scores.argmax(-1) == batch["labels"])).long().cumprod(-1)
            totals["prefix_tokens"] += (valid_correct*batch["valid"]).sum().item()
            indices = torch.linspace(0, len(x)-1, min(32,len(x))).long()
            totals["feature"] += mapper.feature_loss(x[indices].to("cuda"),y[indices].to("cuda")).mean().item()
            totals["blocks"] += 1
    return {key: value/totals["blocks"] if key != "blocks" else value for key,value in totals.items()}


@torch.no_grad()
def export(mapper, work, dest, example_features, draft_state=None):
    source = work/"models/4b/draft"
    state = load_file(str(source/"model.safetensors"))
    if draft_state is not None:
        assert state.keys() == draft_state.keys(), 'Merged drafter key mismatch'
        state = draft_state
    state["fc.weight"] = mapper.folded().detach().cpu().to(torch.bfloat16).contiguous()
    x = example_features[:256].to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = mapper(x)[0]
        actual = mapper.normalize(F.linear(x, state["fc.weight"].to("cuda")))
    relative = ((actual.float()-expected.float()).square().sum(-1) /
                (expected.float().square().sum(-1)+1e-6)).mean().item()
    assert relative < 1e-3, relative
    embedding = tensor(work/"models/4b/target", "model.embed_tokens.weight")
    state["embed_tokens.weight"] = embedding.clone().contiguous()
    state["lm_head.weight"] = embedding.clone().contiguous()
    config = json.loads((source/"config.json").read_text())
    config.update(architectures=["MapperDFlash"], target_hidden_size=4096, tie_word_embeddings=False)
    dest.mkdir(parents=True, exist_ok=True)
    save_file(state, str(dest/"model.safetensors"))
    write(dest/"config.json", config)
    write(dest/"export_check.json", {"context_relative_mse":relative,"tolerance":1e-3,
                                    "preserved_source_embedding_head":True})


def main(args):
    assert torch.cuda.is_available()
    args.out.mkdir(parents=True, exist_ok=True)
    work = Path(os.environ["TRANSFER_WORK"])
    draft, embedding = load_models(work)
    frozen_versions = {name:p._version for name,p in draft.named_parameters()}
    torch.manual_seed(args.seed)
    mapper = LayerContextMapper(draft.fc.weight, draft.hidden_norm.weight).to("cuda")
    train = load_examples(args.data, "train")
    dev = load_examples(args.data, "dev")
    gate = batch_gate(draft, embedding, mapper, train)
    write(args.out/"gate.json", gate)
    print(json.dumps({"gate":gate}), flush=True)
    if args.gate_only:
        return
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=args.lr, weight_decay=0, fused=True)
    randomizer = random.Random(args.seed)
    initial = evaluate(draft, embedding, mapper, dev)
    history = [{"epoch":0,"validation":initial}]
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    steps = 0
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        randomizer.shuffle(order)
        loss_sum = 0.
        active_sum = 0
        for index in order:
            example = train[index]
            row, x, y = example
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.objective == "zip":
                    positions = torch.randperm(len(x))[:32]
                    loss = mapper.feature_loss(x[positions].to("cuda"),y[positions].to("cuda")).mean()
                else:
                    first = len(row["prompt_token_ids"])
                    candidates = list(range(first, len(row["full_ids"])-1))
                    if not candidates:
                        raise ValueError("Record has no supervised block")
                    anchors = randomizer.sample(candidates, min(args.anchors,len(candidates)))
                    batch = collate_blocks([block_for(example,a,draft) for a in anchors], "cuda")
                    result = token_loss(logits(draft, embedding, mapper, batch), batch["labels"], batch["valid"], args.objective)
                    loss = result.loss
                    active_sum += int(result.active.sum())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item()
            steps += 1
        metrics = {"epoch":epoch+1,"mean_training_loss":loss_sum/len(train),"active_labels":active_sum,
                   "validation":evaluate(draft,embedding,mapper,dev),"seconds":time.perf_counter()-started}
        history.append(metrics)
        write(args.out/"history.json", history)
        print(json.dumps(metrics), flush=True)
    assert all(p._version == frozen_versions[n] and p.grad is None for n,p in draft.named_parameters())
    torch.save({"mapper":mapper.state_dict(),"optimizer":optimizer.state_dict(),"seed":args.seed,
                "steps":steps,"objective":args.objective}, args.out/"checkpoint.pt")
    export(mapper, work, args.out/"export", train[0][1])
    write(args.out/"summary.json", {"objective":args.objective,"seed":args.seed,"epochs":args.epochs,
          "records":len(train),"anchors":args.anchors,"microbatch_records":1,"accumulation":1,
          "lr":args.lr,"pilot_only":True,"seconds":time.perf_counter()-started,
          "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"job_id":os.environ.get("SLURM_JOB_ID"),
          "initial":initial,"final":history[-1]["validation"],"frozen_targets":True,
          "checkpoint_sha256":sha(args.out/"checkpoint.pt")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--objective",choices=["auf","ce","zip"],default="auf")
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--anchors",type=int,default=1)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--gate-only",action="store_true")
    main(parser.parse_args())
