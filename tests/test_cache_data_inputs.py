import copy
import hashlib
import json

import pytest

from relayspec.cache_data_inputs import (
    audit_manifest_entries,
    checked_data_inputs,
    require_pilot_data_inputs,
)


def manifests(root, *, interleaved=True):
    files = {}
    for name in ["train-math-2048.json", "train-mixed-2048.json", "validation.json"]:
        labels = (
            ["math"] * 8
            if "train-math" in name
            else ["math", "general_instruction"] * 4
            if interleaved
            else ["math"] * 4 + ["general_instruction"] * 4
        )
        payload = json.dumps({"records": [{"domain": label} for label in labels]})
        (root / name).write_text(payload)
        files[name] = {"sha256": hashlib.sha256(payload.encode()).hexdigest()}
    (root / "manifest-gate.json").write_text(
        json.dumps({"status": "pass", "files": files})
    )


def test_pilot_binds_manifest_and_domain_policy_but_allows_more_records(tmp_path):
    manifests(tmp_path)
    kwargs = {"report_domains": True, "validation_records": 4}
    pilot_inputs = checked_data_inputs(
        tmp_path, "train-mixed-2048.json", train_records=4, **kwargs
    )
    full = checked_data_inputs(
        tmp_path,
        "train-mixed-2048.json",
        train_records=8,
        validation_records=8,
        report_domains=True,
    )
    assert pilot_inputs == full
    require_pilot_data_inputs({"data_inputs": pilot_inputs}, full, custom_inputs=True)
    other_arm = checked_data_inputs(
        tmp_path, "train-math-2048.json", train_records=8, **kwargs
    )
    with pytest.raises(ValueError, match="differ"):
        require_pilot_data_inputs(
            {"data_inputs": pilot_inputs}, other_arm, custom_inputs=True
        )
    changed = copy.deepcopy(full)
    changed["report_domain_diagnostics"] = False
    with pytest.raises(ValueError, match="differ"):
        require_pilot_data_inputs(
            {"data_inputs": pilot_inputs}, changed, custom_inputs=True
        )
    with pytest.raises(ValueError, match="exact data inputs"):
        require_pilot_data_inputs({}, full, custom_inputs=True)
    require_pilot_data_inputs({}, full, custom_inputs=False)


def test_reject_missing_domain_prefix_hash_mismatch_and_oversized_cache(tmp_path):
    manifests(tmp_path, interleaved=False)
    with pytest.raises(ValueError, match="prefix"):
        checked_data_inputs(
            tmp_path,
            "train-mixed-2048.json",
            train_records=4,
            validation_records=4,
            report_domains=True,
        )
    with pytest.raises(ValueError, match="record count"):
        checked_data_inputs(
            tmp_path, "train-math-2048.json", train_records=12, validation_records=4
        )
    (tmp_path / "train-math-2048.json").write_text('{"records": []}')
    with pytest.raises(ValueError, match="hash mismatch"):
        checked_data_inputs(
            tmp_path, "train-math-2048.json", train_records=4, validation_records=4
        )


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "record",
        "label",
        "missing",
        "duplicate",
        "tokens",
        "total",
        "dimensions",
        "file",
    ],
)
def test_cache_index_binds_records_and_actual_domain_token_counts(damage):
    groups = {
        "train": [{"domain": "math", "text": "a"}, {"domain": "general", "text": "b"}]
    }
    entries = [
        {
            "split": "train",
            "index": i,
            "file": f"train/{i:06d}.pt",
            "domain": r["domain"],
            "record_sha256": hashlib.sha256(
                json.dumps(r, sort_keys=True).encode()
            ).hexdigest(),
            "input_width": 6,
            "output_width": 2,
            "tokens": 3 + i,
            "bytes": 100,
            "sha256": "a" * 64,
        }
        for i, r in enumerate(groups["train"])
    ]
    index = {
        "status": "pass",
        "counts": {"train": 2},
        "entries": entries,
        "total_tokens": 7,
        "total_bytes": 200,
        "metadata": {
            "target_hidden_size": 3,
            "target_layer_ids": [1, 2],
            "draft_hidden_size": 2,
            "relay_training": {"max_length": 8},
        },
    }
    if damage == "record":
        groups["train"].reverse()
    elif damage == "label":
        entries[0]["domain"] = "general"
    elif damage == "missing":
        entries.pop()
    elif damage == "duplicate":
        entries[1] = entries[0]
    elif damage == "tokens":
        entries[0]["tokens"] = 9
    elif damage == "total":
        index["total_tokens"] += 1
    elif damage == "dimensions":
        entries[0]["input_width"] = 9
    elif damage == "file":
        entries[0]["file"] = "../train/000000.pt"
    if damage is None:
        result = audit_manifest_entries(index, groups)
        assert result["train"]["by_domain"]["math"]["tokens"] == 3
        assert result["train"]["by_domain"]["general"]["tokens"] == 4
        assert result["train"]["records"] == 2
    else:
        with pytest.raises(ValueError):
            audit_manifest_entries(index, groups)
