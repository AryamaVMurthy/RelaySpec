"""Validation for exact-token, single-turn sequence-scaling inputs."""

import hashlib
import json


def token_ids_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def validate_exact_input(record, *, vocab_size, max_positions, output_cap):
    ids = record["input_ids"]
    if (
        not isinstance(ids, list)
        or not ids
        or any(type(i) is not int or not 0 <= i < vocab_size for i in ids)
    ):
        raise ValueError("invalid exact token input")
    if (
        len(ids) != record["context_tokens"]
        or token_ids_digest(ids) != record["input_ids_sha256"]
    ):
        raise ValueError("exact input length or hash differs")
    if len(ids) + output_cap + 32 > max_positions:
        raise ValueError("input and output exceed model position limit")
    if "turns" in record:
        raise ValueError("exact token input requires a single turn")
    return ids
