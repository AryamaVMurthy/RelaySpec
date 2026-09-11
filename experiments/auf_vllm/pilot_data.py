"""Immutable pilot manifests and dense capture; all teacher calls use vLLM."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def prepare(work, out):
    train = []
    sources = []
    for path in sorted((work / "common/rollouts/train").glob("*.jsonl")):
        train.extend(rows(path)[:32-len(train)])
        sources.append({"path": str(path), "sha256": sha(path)})
        if len(train) == 32:
            break
    dev_path = work / "common/dataset/dev.jsonl"
    dev = rows(dev_path)[:8]
    assert len(train) == 32 and len(dev) == 8
    assert not ({x["group_id"] for x in train} & {x["group_id"] for x in dev})
    for x in train:
        assert x["full_ids"] == x["prompt_token_ids"] + x["output_ids"]
        assert x["temperature"] == 0 and x["thinking"] is False
    for name, data in [("train", train), ("dev-prompts", dev)]:
        dest = out / (name + ".json")
        if dest.exists():
            assert json.loads(dest.read_text()) == data
        else:
            write(dest, data)
    write(out / "provenance.json", {"train_sources": sources, "dev_source": str(dev_path),
          "dev_sha256": sha(dev_path), "teacher": str((work/"models/8b/target").resolve()),
          "target_adapters": None, "training_output_cap": 4096, "pilot_only": True})


def generate_dev(work, out):
    from vllm import LLM, SamplingParams
    dest = out / "dev.json"
    if dest.exists():
        raise FileExistsError("Refusing to replace existing pilot teacher rollouts")
    inputs = json.loads((out / "dev-prompts.json").read_text())
    model = LLM(model=str(work/"models/8b/target"), dtype="bfloat16", enforce_eager=True,
                max_model_len=5120, max_num_seqs=8, max_num_batched_tokens=8192,
                gpu_memory_utilization=.8, enable_prefix_caching=False,
                attention_config={"backend": "FLASH_ATTN"}, seed=42)
    start = time.perf_counter()
    outputs = model.generate([{"prompt_token_ids": x["prompt_token_ids"]} for x in inputs],
                             SamplingParams(temperature=0, max_tokens=4096), use_tqdm=False)
    results = []
    for x, response in zip(inputs, outputs):
        tokens = list(response.outputs[0].token_ids)
        results.append({"row_id": x["row_id"], "group_id": x["group_id"],
                        "prompt_token_ids": x["prompt_token_ids"], "output_ids": tokens,
                        "full_ids": x["prompt_token_ids"] + tokens,
                        "finish_reason": response.outputs[0].finish_reason,
                        "temperature": 0, "thinking": False})
    write(dest, results)
    write(out/"dev-generation.json", {"seconds": time.perf_counter()-start,
          "output_tokens": sum(len(x["output_ids"]) for x in results),
          "job_id": os.environ.get("SLURM_JOB_ID"), "sha256": sha(dest)})


def capture(work, out, size):
    import torch
    from vllm import LLM, PoolingParams
    assert os.environ.get("AUF_CAPTURE") == "1"
    model = LLM(model=str(work/f"models/{size}b/target"), runner="pooling",
                hf_overrides={"architectures": ["AUFCaptureQwen3"]},
                pooler_config={"task": "token_embed"}, dtype="bfloat16", enforce_eager=True,
                max_model_len=5120, max_num_seqs=4, max_num_batched_tokens=8192,
                gpu_memory_utilization=.8, enable_chunked_prefill=False,
                enable_prefix_caching=False, attention_config={"backend": "FLASH_ATTN"})
    for split in ["train", "dev"]:
        manifest = out/f"{split}.json"
        examples = json.loads(manifest.read_text())
        for index, row in enumerate(examples):
            dest = out/f"features/{size}/{split}/{index:05d}.pt"
            meta = dest.with_suffix(".json")
            if dest.exists():
                saved = json.loads(meta.read_text())
                assert saved["sha256"] == sha(dest) and saved["manifest_sha256"] == sha(manifest)
                continue
            start = time.perf_counter()
            result = model.encode([{"prompt_token_ids": row["full_ids"]}], pooling_task="token_embed",
                                  pooling_params=PoolingParams(task="token_embed"), use_tqdm=False)[0]
            features = result.outputs.data.to(torch.bfloat16).cpu()
            assert features.shape == (len(row["full_ids"]), 5*{4:2560, 8:4096}[size])
            assert torch.isfinite(features).all()
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix(".part")
            torch.save({"features": features, "group_id": row["group_id"],
                        "layers": [1,9,17,25,33]}, temp)
            temp.replace(dest)
            write(meta, {"sha256": sha(dest), "manifest_sha256": sha(manifest),
                         "seconds": time.perf_counter()-start, "tokens": len(features),
                         "job_id": os.environ.get("SLURM_JOB_ID")})
            print(json.dumps({"capture": str(dest), "tokens": len(features)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "dev", "capture"])
    parser.add_argument("--size", type=int, choices=[4,8], default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    work = Path(os.environ["TRANSFER_WORK"])
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        prepare(work, args.out)
    elif args.mode == "dev":
        generate_dev(work, args.out)
    else:
        capture(work, args.out, args.size)
