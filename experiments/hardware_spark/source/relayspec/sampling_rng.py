"""Sampling seeds independent of worker sharding and method execution order."""

import hashlib
import json
import math

import torch


def initialize_sampling_rng(temperature, seed_base, problem_id, repetition, turn):
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Sampling temperature must be finite and nonnegative")
    if temperature < 1e-5:
        return None
    if type(seed_base) is not int or seed_base < 0:
        raise ValueError(
            "Sampled evaluation requires explicit nonnegative sampling_seed_base"
        )
    payload = json.dumps(
        [seed_base, str(problem_id), repetition, turn], separators=(",", ":")
    )
    seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") & (
        (1 << 63) - 1
    )
    # torch.manual_seed initializes both CPU and every visible CUDA generator.
    # Each benchmark worker sees one GPU. Reset occurs outside timed generation.
    torch.manual_seed(seed)
    return seed
