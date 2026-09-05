"""Freeze GSM8K confirmation requests after explicit exposure/overlap exclusions."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.ipc as ipc

from relayspec.confirmation_data import CandidateOverlapIndex
from relayspec.scaling_data import OverlapIndex

REVISION = "740312add88f781978c0658806c59bc2815b9866"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"refusing to change an existing frozen artifact: {path}")
    path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrow", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if REVISION not in args.arrow.parts or args.arrow.name != "gsm8k-test.arrow":
        raise ValueError("expected the pinned GSM8K main test Arrow cache")
    with args.arrow.open("rb") as stream:
        rows = ipc.open_stream(stream).read_all().to_pylist()
    if len(rows) != 1319 or any(not {"question", "answer"} <= r.keys() for r in rows):
        raise ValueError("unexpected pinned GSM8K test schema/count")
    candidates = []
    for row in rows:
        if "####" not in row["answer"]:
            raise ValueError("GSM8K reference lacks the declared answer delimiter")
        candidates.append(
            {
                "benchmark": "gsm8k",
                "problem_id": "gsm8k/"
                + hashlib.sha256(row["question"].encode()).hexdigest()[:16],
                "prompt": row["question"],
                "answer": row["answer"].split("####")[-1].strip().replace(",", ""),
            }
        )
    if len({r["problem_id"] for r in candidates}) != len(candidates):
        raise ValueError("duplicate candidate IDs require explicit resolution")
    index = CandidateOverlapIndex([r["prompt"] for r in candidates])
    exclusions = {}
    inputs = {str(args.arrow): digest(args.arrow)}
    corpus_counts = Counter()

    def exclude(text, source):
        corpus_counts[source] += 1
        for i, reason in index.matches(text).items():
            exclusions.setdefault(i, {"source": source, "match": reason})

    manifest_gate = json.loads((args.manifest_root / "manifest-gate.json").read_text())
    if manifest_gate["status"] != "pass":
        raise ValueError("fitting manifest audit did not pass")
    for name in ("train-32768.json", "validation.json"):
        path = args.manifest_root / name
        if digest(path) != manifest_gate["files"][name]["sha256"]:
            raise ValueError("fitting manifest hash changed")
        inputs[str(path)] = digest(path)
        for record in json.loads(path.read_text())["records"]:
            for key in ("problem", "solution"):
                exclude(record[key], name)
    historical = Path("configs/train_math_4096.json")
    inputs[str(historical)] = digest(historical)
    for record in json.loads(historical.read_text())["records"]:
        for key in ("problem", "solution"):
            exclude(record[key], str(historical))
    known_ids = set()
    for path in sorted(Path("configs").glob("eval_manifest*.json")):
        inputs[str(path)] = digest(path)
        for record in json.loads(path.read_text())["records"]:
            if record["benchmark"] == "gsm8k":
                known_ids.add(record["problem_id"])
            if "prompt" in record:
                exclude(record["prompt"], "existing_evaluation_manifests")
    raw_files = sorted((args.raw_root / "reports").rglob("benchmark-rank*.jsonl"))
    if not raw_files:
        raise ValueError("no collected evaluation history found")
    for path in raw_files:
        inputs[str(path)] = digest(path)
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("benchmark") == "gsm8k":
                    known_ids.add(record["problem_id"])
    for i, record in enumerate(candidates):
        if record["problem_id"] in known_ids:
            exclusions.setdefault(
                i, {"source": "recorded_evaluation_id", "match": "id"}
            )
    eligible = [r for i, r in enumerate(candidates) if i not in exclusions]
    eligible.sort(
        key=lambda r: hashlib.sha256(
            ("20260906:" + r["problem_id"]).encode()
        ).hexdigest()
    )
    unique, internal = [], OverlapIndex()
    for record in eligible:
        if internal.match(record["prompt"]):
            continue
        internal.add(record["prompt"])
        unique.append(record)
    if len(unique) < 512:
        raise ValueError(
            "insufficient independent candidate requests for confirmation/reserve"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, records in [("confirmation", unique[:256]), ("reserve", unique[256:512])]:
        path = args.output_dir / f"gsm8k-{name}-256.json"
        freeze(
            path,
            {
                "dataset_revisions": {"gsm8k": REVISION},
                "seed": 20260906,
                "records": records,
            },
        )
        outputs[path.name] = digest(path)
    freeze(
        args.output_dir / "confirmation-data-gate.json",
        {
            "status": "pass",
            "dataset": "openai/gsm8k",
            "subset": "main",
            "split": "test",
            "revision": REVISION,
            "source_records": len(candidates),
            "exposed_gsm8k_ids": len(known_ids),
            "scanned_rank_files": len(raw_files),
            "excluded_candidates": len(exclusions),
            "eligible_after_corpus_exclusion": len(eligible),
            "eligible_after_internal_exclusion": len(unique),
            "corpus_text_counts": dict(corpus_counts),
            "exclusion_details": {
                candidates[i]["problem_id"]: reason
                for i, reason in sorted(exclusions.items())
            },
            "input_sha256": inputs,
            "output_sha256": outputs,
            "builder_sha256": digest(Path(__file__)),
            "helper_sha256": digest(Path("src/relayspec/confirmation_data.py")),
            "lexical_rule": "Normalized exact text; 5-shingle Jaccard >=0.6 for texts with >=10 unique shingles. Check every fitting problem/solution and recorded evaluation prompt. Also exclude recorded GSM8K IDs and pairwise overlap within confirmation/reserve.",
            "scope": "Unexposed relative to the scanned local manifests and collected rank logs, not a guarantee of global non-exposure or semantic/template independence. Reading the existing 32768-record manifest is an exclusion audit and creates no new fitting run. Freeze selected model/settings and quality margin before generating/scoring confirmation outputs. Reserve requests must not be used for tuning.",
        },
    )
    print(
        json.dumps(
            {
                "excluded": len(exclusions),
                "eligible": len(unique),
                "confirmation": 256,
                "reserve": 256,
            }
        )
    )


if __name__ == "__main__":
    main()
