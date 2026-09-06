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
