"""Bind cache extraction to checked manifests and the pilot's exact data inputs."""

import hashlib
import json
from collections import Counter
from pathlib import Path


def checked_data_inputs(
    root, training_manifest, *, train_records, validation_records, report_domains=False
):
    root = Path(root)
    if Path(training_manifest).name != training_manifest:
        raise ValueError("training manifest must be a filename within the data root")
    gate_bytes = (root / "manifest-gate.json").read_bytes()
    gate = json.loads(gate_bytes)
    if gate.get("status") != "pass":
        raise ValueError("data manifest gate did not pass")
    result = {
        "manifest_gate_sha256": hashlib.sha256(gate_bytes).hexdigest(),
        "report_domain_diagnostics": report_domains,
        "files": {},
    }
    for split, name, count in [
        ("train", training_manifest, train_records),
        ("validation", "validation.json", validation_records),
    ]:
        payload = (root / name).read_bytes()
        sha = hashlib.sha256(payload).hexdigest()
        if sha != gate["files"][name]["sha256"]:
            raise ValueError(f"manifest hash mismatch: {name}")
        records = json.loads(payload)["records"]
        if not isinstance(count, int) or count < 4 or count > len(records):
            raise ValueError(f"invalid requested cache record count: {name}")
        item = {"name": name, "sha256": sha, "available_records": len(records)}
        if report_domains:
            labels = [r.get("domain") for r in records]
            if any(not isinstance(v, str) or not v for v in labels):
                raise ValueError(f"domain diagnostics require labeled records: {name}")
            item["domain_counts"] = dict(sorted(Counter(labels).items()))
            if set(labels[:count]) != set(labels):
                raise ValueError(
                    f"cache prefix does not cover declared domains: {name}"
                )
        result["files"][split] = item
    return result


def require_pilot_data_inputs(pilot, inputs, *, custom_inputs):
    """Old default Numina pilots remain valid; new/custom inputs must be bound."""
    prior = pilot.get("data_inputs")
    if prior is None and custom_inputs:
        raise ValueError("custom extraction requires a pilot with exact data inputs")
    if prior is not None and prior != inputs:
        raise ValueError("extraction data inputs differ from the successful pilot")


def audit_manifest_entries(index, groups):
    """Bind copied cache-index records to exact manifest prefixes and domain counts.

    This checks index provenance/accounting only. GPU consumers separately verify
    the hashes of actual feature tensors before fitting.
    """
    counts = {split: len(records) for split, records in groups.items()}
    expected = {(split, i) for split, count in counts.items() for i in range(count)}
    entries = index["entries"]
    if (
        index.get("status") != "pass"
        or index["counts"] != counts
        or len(entries) != len(expected)
        or {(e["split"], e["index"]) for e in entries} != expected
    ):
        raise ValueError("cache index does not cover exact manifest prefixes")
    metadata = index["metadata"]
    input_width = metadata["target_hidden_size"] * len(metadata["target_layer_ids"])
    output_width = metadata["draft_hidden_size"]
    maximum = metadata["relay_training"]["max_length"]
    report = {
        split: {"records": 0, "tokens": 0, "bytes": 0, "by_domain": {}}
        for split in groups
    }
    for entry in entries:
        split, i = entry["split"], entry["index"]
        record = groups[split][i]
        sha = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        domain = record.get("domain")
        if (
            entry["record_sha256"] != sha
            or entry.get("domain") != domain
            or not isinstance(domain, str)
            or not domain
            or entry["input_width"] != input_width
            or entry["output_width"] != output_width
            or entry["file"] != f"{split}/{i:06d}.pt"
            or not isinstance(entry["tokens"], int)
            or not 0 < entry["tokens"] <= maximum
            or not isinstance(entry["bytes"], int)
            or entry["bytes"] <= 0
            or not isinstance(entry["sha256"], str)
            or len(entry["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in entry["sha256"])
        ):
            raise ValueError(
                "cache entry differs from its manifest record or tensor metadata"
            )
        group = report[split]
        subgroup = group["by_domain"].setdefault(
            domain, {"records": 0, "tokens": 0, "bytes": 0}
        )
        for key, amount in [
            ("records", 1),
            ("tokens", entry["tokens"]),
            ("bytes", entry["bytes"]),
        ]:
            group[key] += amount
            subgroup[key] += amount
    if (
        sum(g["tokens"] for g in report.values()) != index["total_tokens"]
        or sum(g["bytes"] for g in report.values()) != index["total_bytes"]
    ):
        raise ValueError("cache index totals do not reproduce from entries")
    return report
