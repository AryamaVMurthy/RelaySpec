"""Shuffle chunks and their records while bounding feature-metadata churn."""
import random


def epoch_order(records, epoch, seed=42, chunk_size=64):
    if records <= 0 or chunk_size <= 0 or records % chunk_size:
        raise ValueError('Full chunks required')
    rng = random.Random(seed + epoch)
    chunks = list(range(records // chunk_size))
    rng.shuffle(chunks)
    result = []
    for chunk in chunks:
        rows = list(range(chunk * chunk_size, (chunk + 1) * chunk_size))
        rng.shuffle(rows)
        result.extend(rows)
    return result
