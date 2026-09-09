"""Explicit data budgets and snapshots for controlled fitting experiments."""

import hashlib
import json
from pathlib import Path

import torch


def training_data_accounting(records, training, world_size):
    if not records or len(records) % world_size:
        raise ValueError("training records must form nonempty equal rank shards")
    identities = [
        r.get("problem_sha256") or hashlib.sha256(r["problem"].encode()).hexdigest()
        for r in records
    ]
    distinct = len(set(identities))
    expected = training.get("distinct_examples")
    if expected is not None and (distinct != int(expected) or len(records) != distinct):
        raise ValueError("controlled manifest must contain exactly the distinct budget")
    return {
        "manifest_records": len(records),
        "distinct_records": distinct,
        "record_presentations": int(training["steps"]) * world_size,
        "records_per_update": world_size,
        "ordered_ids_sha256": hashlib.sha256(
            json.dumps(identities).encode()
        ).hexdigest(),
    }


def save_training_checkpoint(path, payload, optimizer, elapsed_training_seconds):
    """Save without touching the live optimizer or advancing any RNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    snapshot = {
        **payload,
        "optimizer": optimizer.state_dict(),
        "elapsed_training_seconds": elapsed_training_seconds,
        "optimizer_continuity": "one uninterrupted training process",
    }
    temporary = path.with_suffix(".tmp")
    torch.save(snapshot, temporary)
    temporary.replace(path)
