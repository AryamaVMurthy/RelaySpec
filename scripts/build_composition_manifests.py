"""Build matched2048-record math/mixed pools with common domain validation."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from relayspec.scaling_data import OverlapIndex, content_hash, stratified_order


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_path = Path("reports/mapper-scaling-20260905/composition-data/source.json")
    source = json.loads(source_path.read_text())
    raw_path = Path(source["files"]["databricks-dolly-15k.jsonl"]["path"])
    if (
        source["revision"] != "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
        or digest(raw_path)
        != "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
    ):
        raise ValueError("general-instruction source differs from pinned input")
    math_gate_path = Path("reports/mapper-scaling-20260905/data/manifest-gate.json")
    math_gate = json.loads(math_gate_path.read_text())
    math_root = Path("data/scaling/manifests-pinned")
    pools = {}
    inputs = [source_path, raw_path, math_gate_path]
    for name in ["train-2048.json", "validation.json"]:
        path = math_root / name
        if digest(path) != math_gate["files"][name]["sha256"]:
            raise ValueError("math fitting pool changed")
        pools[name] = json.loads(path.read_text())["records"]
        inputs.append(path)
    references = [
        Path("configs/eval_manifest.json"),
        Path("configs/eval_manifest_full_v4.json"),
        Path("configs/train_math_4096.json"),
        Path(
            "configs/submission/confirmation-gsm8k-20260906/gsm8k-confirmation-256.json"
        ),
        Path("configs/submission/confirmation-gsm8k-20260906/gsm8k-reserve-256.json"),
    ]
    confirmation = json.loads(
        Path("configs/submission/confirmation-gsm8k-20260906/protocol.json").read_text()
    )
    for path in references[-2:]:
        if digest(path) != confirmation["data_sha256"][path.name]:
            raise ValueError("confirmation overlap-exclusion input changed")
    math_test = Path("data/scaling/math-reference/data/test-00000-of-00001.parquet")
    if (
        digest(math_test)
        != "7dca8d6e41af88ecf82f2b5f36eb5530e083aaaa86ee325f62bd5c31535178c6"
    ):
        raise ValueError("complete MATH reference test changed")
    reference_rows = pq.read_table(math_test).to_pylist()
    for path in references:
        reference_rows.extend(json.loads(path.read_text())["records"])
    reference_rows += pools["train-2048.json"] + pools["validation.json"]
    inputs += references + [math_test]
    index, seen_reference_text = OverlapIndex(0.6), set()
    for row in reference_rows:
        for key in ["problem", "prompt", "solution"]:
            text = row.get(key)
            if text and content_hash(text) not in seen_reference_text:
                index.add(text)
                seen_reference_text.add(content_hash(text))
    candidates = []
    for line_number, line in enumerate(raw_path.read_text().splitlines()):
        row = json.loads(line)
        problem = row["instruction"].strip()
        if row["context"].strip():
            problem += "\n\n" + row["context"].strip()
        if not problem or not row["response"].strip() or not row["category"]:
            continue
        candidates.append(
            {
                "problem": problem,
                "solution": row["response"].strip(),
                "source": "dolly:" + row["category"],
                "category": row["category"],
                "source_row": line_number,
                "domain": "general_instruction",
            }
        )
    exclusions, seen = Counter(), set()
    general_validation, general_training = [], []
    for row in stratified_order(candidates, 1729):
        key = content_hash(row["problem"])
        if key in seen:
            exclusions["duplicate_instruction_context"] += 1
            continue
        seen.add(key)
        match = index.match(row["problem"]) or index.match(row["solution"])
        if match:
            exclusions[match] += 1
            continue
        row.update(
            normalized_problem_sha256=key,
            problem_sha256=hashlib.sha256(row["problem"].encode()).hexdigest(),
            solution_sha256=hashlib.sha256(row["solution"].encode()).hexdigest(),
        )
        if len(general_validation) < 1024:
            general_validation.append(row)
            index.add(row["problem"])
            index.add(row["solution"])
        else:
            general_training.append(row)
        if len(general_training) == 1024:
            break
    if len(general_training) != 1024 or len(general_validation) != 1024:
        raise ValueError(
            "insufficient distinct general records after overlap filtering"
        )
    math_training = [{**row, "domain": "math"} for row in pools["train-2048.json"]]
    math_validation = [{**row, "domain": "math"} for row in pools["validation.json"]]
    mixed_training = [
        row
        for pair in zip(math_training[:1024], general_training, strict=True)
        for row in pair
    ]
    # Exercise both domains even in the bounded 16-record cache pilot.
    common_validation = [
        row
        for pair in zip(math_validation, general_validation, strict=True)
        for row in pair
    ]
    train_keys = {content_hash(r["problem"]) for r in math_training + general_training}
    if train_keys & {content_hash(r["problem"]) for r in common_validation}:
        raise ValueError("training and validation overlap")
    args.output.mkdir(parents=True)
    (args.output / "ATTRIBUTION.md").write_text(
        "General-instruction records are adapted from Databricks Dolly 15k.\n\n"
        "Source: https://huggingface.co/datasets/databricks/databricks-dolly-15k\n\n"
        "License: CC BY-SA 3.0, https://creativecommons.org/licenses/by-sa/3.0/\n\n"
        f"Pinned revision: {source['revision']}\n\n"
        "Changes: deterministic subset selection and lexical overlap filtering; "
        "instruction and context joined, metadata added, and selected records "
        "combined with the separately sourced Numina math pool.\n"
    )
    files = {}
    for name, records, role in [
        ("train-math-2048.json", math_training, "math-only fitting"),
        (
            "train-mixed-2048.json",
            mixed_training,
            "half math / half general-instruction fitting",
        ),
        (
            "validation.json",
            common_validation,
            "common fitting validation,1024 per domain",
        ),
    ]:
        if (
            len(records) != 2048
            or len({content_hash(r["problem"]) for r in records}) != 2048
        ):
            raise ValueError("composition pool is not exactly2048 distinct examples")
        counts = dict(Counter(r["domain"] for r in records))
        payload = {
            "role": role,
            "records": records,
            "distinct_records": len(records),
            "domain_counts": counts,
            "source_counts": dict(Counter(r["source"] for r in records)),
            "total_characters": sum(
                len(r["problem"]) + len(r["solution"]) for r in records
            ),
            "attribution": "General-instruction records from Databricks databricks-dolly-15k, CC-BY-SA3.0, pinned revision"
            + source["revision"],
        }
        path = args.output / name
        path.write_text(json.dumps(payload, indent=2) + "\n")
        files[name] = {
            "sha256": digest(path),
            "records": len(records),
            "domain_counts": counts,
            "source_counts": payload["source_counts"],
            "total_characters": payload["total_characters"],
        }
    gate = {
        "status": "pass",
        "files": files,
        "seed": 1729,
        "validation_order": "alternating_math_general_instruction_v2",
        "general_source": source,
        "input_sha256": {str(p): digest(p) for p in inputs},
        "exclusions": dict(exclusions),
        "reference_unique_texts": len(seen_reference_text),
        "builder_sha256": digest(Path(__file__)),
        "helper_sha256": digest(Path("src/relayspec/scaling_data.py")),
        "scope": "Matched2048 distinct fitting records, common2048 validation records. General-instruction source may contain math. Exact normalized and5-shingle Jaccard>=0.6 exclusion for texts with>=10 shingles. Does not prove semantic/template independence or absence from pretraining. Confirmation/reserve text used only for deterministic overlap exclusion, without generation, outcome inspection or prompt-specific tuning. Equal record/update counts do not equalize non-padding tokens; report actual token costs after extraction.",
    }
    (args.output / "manifest-gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps({"files": files, "exclusions": dict(exclusions)}))


if __name__ == "__main__":
    main()
