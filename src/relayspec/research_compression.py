"""Activation-weighted projection compression for bounded research screens."""

import hashlib
import json
from pathlib import Path

import torch

from relayspec.feature_cache import load_cached_record


def activation_compression(base, cache_path, ranks, seed=1729):
    root = Path(cache_path)
    index_path = root / "cache-index.json"
    index = json.loads(index_path.read_text())
    meta = index["metadata"]
    if (
        index["status"] != "pass"
        or base["target_layer_ids"] != meta["target_layer_ids"]
    ):
        raise ValueError(
            "Activation compression requires an exact matching target-tap cache"
        )
    w = base["relay"]["projection.weight"].float().cuda()
    generator = torch.Generator().manual_seed(seed)

    def sample(split, count):
        entries = [e for e in index["entries"] if e["split"] == split][:count]
        if len(entries) != count:
            raise ValueError("Insufficient cached records")
        samples = []
        for e in entries:
            x, _ = load_cached_record(root, e)
            x = x.reshape(-1, w.shape[1])
            chosen = torch.randperm(x.shape[0], generator=generator)[:8]
            samples.append(x[chosen].float())
        x = torch.cat(samples).cuda()
        if base.get("relay_architecture", "normalized_linear") == "normalized_linear":
            x = x * torch.rsqrt(
                x.square().mean(-1, keepdim=True) + meta["rms_norm_eps"]
            )
        return x

    train = sample("train", 512)
    validation = sample("validation", 128)
    with torch.no_grad():
        y = train @ w.T
        val_y = validation @ w.T
        torch.manual_seed(seed)
        _, _, basis = torch.svd_lowrank(
            y, q=min(max(ranks) + 32, min(y.shape)), niter=4
        )
        outputs = {}
        errors = {}
        for rank in ranks:
            left = basis[:, :rank].contiguous()
            right = (left.T @ w).contiguous()
            outputs[rank] = {
                "projection.0.weight": right.cpu(),
                "projection.1.weight": left.cpu(),
            }
            predicted = (validation @ right.T) @ left.T
            errors[str(rank)] = float(
                (predicted - val_y).square().sum() / val_y.square().sum()
            )
    return outputs, dict(
        cache_index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
        train_records=512,
        validation_records=128,
        tokens_per_record_at_most=8,
        seed=seed,
        sampled_train_tokens=len(train),
        sampled_validation_tokens=len(validation),
        heldout_relative_projection_mse=errors,
        scope="PCA of uncentered mapped training activations, followed by output-subspace projection of W. No drafter update. Errors are pre-output-normalization projection MSE; validation is fitting validation, not fresh decoding data.",
    )
