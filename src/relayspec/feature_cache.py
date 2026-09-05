"""Lossless frozen-feature storage and record-weighted masked mapper batches."""

import hashlib
import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def write_cache_shard(root, groups, *, rank, world_size, extract):
    root = Path(root)
    entries = []
    for split, records in groups.items():
        (root / split).mkdir(parents=True, exist_ok=True)
        for index in range(rank, len(records), world_size):
            row = records[index]
            x, y, ids = extract(row)
            payload = {
                "x": x.detach().cpu().contiguous(),
                "y": y.detach().cpu().contiguous(),
                "input_ids": ids.detach().cpu().contiguous(),
            }
            if x.shape[:2] != y.shape[:2] or x.shape[0] != 1 or x.shape[1] < 1:
                raise ValueError(
                    "cache requires paired nonempty unpadded individual records"
                )
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError("nonfinite frozen features")
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            data = buffer.getvalue()
            # Test serialization without recomputing or quantizing features.
            if index == rank:
                restored = torch.load(io.BytesIO(data), weights_only=True)
                if any(not torch.equal(payload[k], restored[k]) for k in payload):
                    raise ValueError("cache round-trip changed a tensor")
            relative = f"{split}/{index:06d}.pt"
            with (root / relative).open("xb") as handle:
                handle.write(data)
            entries.append(
                {
                    "split": split,
                    "index": index,
                    "file": relative,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "tokens": x.shape[1],
                    "input_width": x.shape[2],
                    "output_width": y.shape[2],
                    "record_sha256": hashlib.sha256(
                        json.dumps(row, sort_keys=True).encode()
                    ).hexdigest(),
                }
            )
    (root / f"cache-rank{rank}.json").write_text(json.dumps(entries, indent=2) + "\n")
    return entries


def finalize_cache(root, *, counts, world_size, metadata):
    root = Path(root)
    entries = [
        entry
        for rank in range(world_size)
        for entry in json.loads((root / f"cache-rank{rank}.json").read_text())
    ]
    expected = {(split, i) for split, n in counts.items() for i in range(n)}
    if {(e["split"], e["index"]) for e in entries} != expected or len(entries) != len(
        expected
    ):
        raise ValueError("cache shards do not cover the exact declared records")
    if len({(e["input_width"], e["output_width"]) for e in entries}) != 1:
        raise ValueError("cache feature dimensions changed between records")
    for entry in entries:
        if (root / entry["file"]).stat().st_size != entry["bytes"]:
            raise ValueError("cache file size differs from recorded bytes")
    result = {
        "status": "pass",
        "metadata": metadata,
        "counts": counts,
        "total_bytes": sum(e["bytes"] for e in entries),
        "total_tokens": sum(e["tokens"] for e in entries),
        "entries": sorted(entries, key=lambda e: (e["split"], e["index"])),
    }
    (root / "cache-index.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def load_cached_record(root, entry):
    data = (Path(root) / entry["file"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError("cached feature hash mismatch")
    payload = torch.load(io.BytesIO(data), weights_only=True)
    return payload["x"], payload["y"]


class MappedFeatureReader:
    """Verify once, retain read-only tensor views, and reject later file changes.

    Feature files are immutable experiment inputs. Keeping their verified views
    avoids reparsing and rehashing the same records on every fitting epoch.
    Unlike a RAM preload, mmap lets the OS reclaim physical pages as needed.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.records = {}

    @staticmethod
    def identity(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def __call__(self, entry):
        path = self.root / entry["file"]
        identity = self.identity(path)
        cached = self.records.get(entry["file"])
        if cached is not None:
            original, expected_hash, tensors = cached
            if identity != original or entry["sha256"] != expected_hash:
                raise ValueError("verified mapped feature file changed")
            return tensors
        if identity[2] != entry["bytes"]:
            raise ValueError("mapped feature file size mismatch")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("mapped feature hash mismatch")
        payload = torch.load(path, mmap=True, weights_only=True, map_location="cpu")
        if self.identity(path) != identity:
            raise ValueError("mapped feature file changed during verification")
        tensors = payload["x"], payload["y"]
        self.records[entry["file"]] = identity, entry["sha256"], tensors
        return tensors


def pad_feature_batch(examples, device):
    if not examples:
        raise ValueError("feature batch must be nonempty")
    lengths = torch.tensor([x.shape[1] for x, _ in examples], device=device)
    if any(x.shape[0] != 1 or x.shape[:2] != y.shape[:2] for x, y in examples):
        raise ValueError("features must be paired individual records")
    x = pad_sequence([x.squeeze(0) for x, _ in examples], batch_first=True).to(device)
    y = pad_sequence([y.squeeze(0) for _, y in examples], batch_first=True).to(device)
    mask = torch.arange(x.shape[1], device=device)[None, :] < lengths[:, None]
    return x, y, mask


def record_weighted_loss(
    prediction, target, mask, *, objective, historical_cosine_weight=None
):
    if prediction.shape != target.shape or prediction.shape[:2] != mask.shape:
        raise ValueError("batch prediction, target and mask shapes must match")
    if mask.dtype != torch.bool or not mask.any(dim=1).all():
        raise ValueError("each record must contain valid tokens")
    prediction = prediction.float()
    target = target.detach().float()
    # Zero padding before squaring/dividing to keep masked positions finite.
    prediction = prediction.masked_fill(~mask[..., None], 0)
    target = target.masked_fill(~mask[..., None], 0)
    squared = (prediction - target).square()
    if objective == "relative_interface_mse":
        energy = target.square().sum(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
        per_token = squared.sum(dim=-1) / energy
    elif objective == "raw_mse":
        per_token = squared.mean(dim=-1)
    elif objective == "historical_mse_cosine":
        if historical_cosine_weight is None:
            raise ValueError("historical cosine weight must be explicit")
        per_token = squared.mean(dim=-1) + historical_cosine_weight * (
            1 - F.cosine_similarity(prediction, target, dim=-1)
        )
    else:
        raise ValueError(f"unknown interface objective: {objective}")
    return (per_token.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1)).mean()


def restore_output_norm(root, metadata, device):
    path = Path(root) / "output-norm.pt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["output_norm_sha256"]:
        raise ValueError("cached output normalization hash mismatch")
    spec = torch.load(path, map_location="cpu", weights_only=True)
    if spec is None:
        if metadata["family"] != "eagle3":
            raise ValueError("DFlash cache is missing its released output norm")
        return torch.nn.Identity().to(device)
    if spec["kind"] != "Qwen3RMSNorm" or metadata["family"] != "dflash":
        raise ValueError("unsupported cached output normalization")
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    norm = Qwen3RMSNorm(len(spec["weight"]), eps=spec["eps"])
    norm.load_state_dict({"weight": spec["weight"]}, strict=True)
    return (
        norm.to(device=device, dtype=spec["weight"].dtype).requires_grad_(False).eval()
    )
