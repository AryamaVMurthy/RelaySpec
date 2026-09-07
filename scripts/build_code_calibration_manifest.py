"""Freeze a Python-instruction calibration pool with explicit lexical exclusions."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from relayspec.scaling_data import OverlapIndex, content_hash


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source_root = Path("data/scaling/codealpaca-source")
    source = json.loads((source_root / "source.json").read_text())
    assert source["revision"] == "2f78ddc5c682ed6738ad092bbbfa59ba915afcb0"
    for item in source["files"].values():
        assert digest(Path(item["path"])) == item["sha256"]
    raw = json.loads((source_root / "code_alpaca_20k.json").read_text())
    index = OverlapIndex(0.6)
    seen = set()
    inputs = {}
    for p in sorted(Path("configs").rglob("*.json")):
        data = json.loads(p.read_text())
        if not isinstance(data, dict) or not isinstance(data.get("records"), list):
            continue
        inputs[str(p)] = digest(p)
        for row in data["records"]:
            for key in ("problem", "prompt", "solution"):
                value = row.get(key)
                if isinstance(value, str) and value and content_hash(value) not in seen:
                    index.add(value)
                    seen.add(content_hash(value))
    candidates = []
    for number, row in enumerate(raw):
        if "python" not in row["instruction"].lower() or not row["output"].strip():
            continue
        problem = row["instruction"].strip()
        if row["input"].strip():
            problem += "\n\n" + row["input"].strip()
        candidates.append(
            dict(
                problem=problem,
                solution=row["output"].strip(),
                source="codealpaca-python-instruction",
                source_row=number,
                domain="code_instruction",
            )
        )
    candidates.sort(
        key=lambda r: hashlib.sha256(
            f"1729:{r['problem']}:{r['solution']}".encode()
        ).hexdigest()
    )
    selected = []
    excluded = Counter()
    for row in candidates:
        match = index.match(row["problem"]) or index.match(row["solution"])
        if match:
            excluded[match] += 1
            continue
        row["problem_sha256"] = hashlib.sha256(row["problem"].encode()).hexdigest()
        row["solution_sha256"] = hashlib.sha256(row["solution"].encode()).hexdigest()
        row["normalized_problem_sha256"] = content_hash(row["problem"])
        selected.append(row)
        index.add(row["problem"])
        index.add(row["solution"])
        if len(selected) == 640:
            break
    assert len(selected) == 640
    root = Path("data/scaling/code-calibration-v1")
    if root.exists():
        raise FileExistsError(root)
    root.mkdir()
    files = {}
    for name, rows in [
        ("validation.json", selected[:128]),
        ("train-512.json", selected[128:]),
    ]:
        p = root / name
        p.write_text(
            json.dumps(
                {
                    "records": rows,
                    "scope": "Calibration text, not evaluation. Python-keyword subset of synthetic CodeAlpaca instructions, no asserted code correctness.",
                },
                indent=2,
            )
            + "\n"
        )
        files[name] = {"sha256": digest(p), "records": len(rows)}
    (root / "ATTRIBUTION.md").write_text(
        "Adapted from Code Alpaca by Sahil Chaudhary (2023).\n\nSource: https://github.com/sahil280114/codealpaca\n\nPinned revision: "
        + source["revision"]
        + "\n\nData license: CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/\n\nChanges: Python-keyword instruction subset, deterministic ordering, lexical overlap filtering, joined instruction/input, split and metadata. Synthetic output correctness is not established.\n"
    )
    (root / "DATA_LICENSE").write_bytes((source_root / "DATA_LICENSE").read_bytes())
    result = dict(
        status="pass",
        source=source,
        files=files,
        reference_input_sha256=inputs,
        reference_unique_texts=len(seen),
        python_keyword_candidates=len(candidates),
        excluded=dict(excluded),
        seed=1729,
        scope="Exact normalized and5-shingle Jaccard>=0.6 exclusions against all record-bearing config manifests and earlier selected calibration texts. Strings with fewer than10 unique shingles use exact exclusion only. This is not proof of semantic or pretraining independence.128 validation and512 training records, disjoint.",
    )
    (root / "manifest-gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "files",
                    "reference_unique_texts",
                    "python_keyword_candidates",
                    "excluded",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
