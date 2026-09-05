"""Build nested fitting sets from an immutable, declared NuminaMath shard."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import relayspec.scaling_data as scaling_data
from relayspec.scaling_data import OverlapIndex, content_hash, stratified_order


def main():
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--math-test", type=Path, required=True)
    parser.add_argument(
        "--evaluation", type=Path, default=Path("configs/eval_manifest.json")
    )
    parser.add_argument(
        "--historical-fit", type=Path, default=Path("configs/train_math_4096.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=32768)
    parser.add_argument("--validation-records", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    expected_sources = {
        args.parquet: "25c0fa2ca96dd8078470ad1bbc439d0d211fd8b7d22e04abf7f61e9be9bf9ec3",
        args.math_test: "7dca8d6e41af88ecf82f2b5f36eb5530e083aaaa86ee325f62bd5c31535178c6",
    }
    for path, expected in expected_sources.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(
                f"input does not match the declared pinned dataset shard: {path}"
            )
    started = time.monotonic()
    # These three sufficiently large strata avoid explicitly named benchmark
    # collections; lexical filtering still runs on every retained problem/solution.
    allowed = {"cn_k12", "synthetic_math", "orca_math"}
    rows = pq.read_table(
        args.parquet, columns=["source", "problem", "solution"]
    ).to_pylist()
    rows = [r for r in rows if r["source"] in allowed]
    reference_rows = pq.read_table(args.math_test).to_pylist()
    reference_rows += json.loads(args.historical_fit.read_text())["records"]
    reference_rows += json.loads(args.evaluation.read_text())["records"]
    index = OverlapIndex(0.6)
    for row in reference_rows:
        for text in (
            row.get("problem", row.get("prompt", "")),
            row.get("solution", ""),
        ):
            if text:
                index.add(text)
    exclusions = Counter()
    seen_problems = set()
    validation = []
    training = []
    for row in stratified_order(rows, args.seed):
        key = content_hash(row["problem"])
        if key in seen_problems:
            exclusions["duplicate_problem"] += 1
            continue
        seen_problems.add(key)
        match = index.match(row["problem"]) or index.match(row["solution"])
        if match:
            exclusions[match] += 1
            continue
        item = {
            **row,
            "problem_sha256": hashlib.sha256(row["problem"].encode()).hexdigest(),
            "solution_sha256": hashlib.sha256(row["solution"].encode()).hexdigest(),
            "normalized_problem_sha256": key,
        }
        if len(validation) < args.validation_records:
            validation.append(item)
            index.add(row["problem"])
            index.add(row["solution"])
        else:
            training.append(item)
        if len(training) == args.max_records:
            break
    if len(training) != args.max_records:
        raise ValueError(
            "eligible unique population does not meet the requested budget"
        )
    provenance = {
        "source": {
            "id": "AI-MO/NuminaMath-CoT",
            "revision": "9d8d210c9f6a36c8f3cd84045668c9b7800ef517",
            "split": "train",
            "shard": "data/train-00000-of-00005.parquet",
            "allowed_sources": sorted(allowed),
        },
        "reference_math": {
            "id": "DigitalLearningGmbH/MATH-lighteval",
            "revision": "0530c78699ea5e8eb5530600900e1f328b48acad",
            "split": "test",
        },
        "seed": args.seed,
        "eligible_source_records_before_audit": len(rows),
        "reference_records": len(reference_rows),
        "exclusions": dict(exclusions),
        "lexical_filter": "Exact normalized problem/solution; 5-shingle Jaccard >= 0.6 for texts with >= 10 unique shingles. Validation also excluded from training. Does not guarantee semantic/template independence.",
        "scope": "Separate expanded-data study; do not pool with historical MATH-only fitting curves. Nested source-stratified prefixes. Fitting-validation is not confirmatory decoding evaluation.",
        "input_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                args.parquet,
                args.math_test,
                args.evaluation,
                args.historical_fit,
            ]
        },
        "helper_sha256": hashlib.sha256(
            Path(scaling_data.__file__).read_bytes()
        ).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.mkdir(parents=True)
    files = {}

    def save(name, records, role):
        path = args.output / name
        payload = {
            "role": role,
            "provenance": provenance,
            "source_counts": dict(Counter(r["source"] for r in records)),
            "records": records,
            "distinct_records": len(records),
            "total_characters": sum(
                len(r["problem"]) + len(r["solution"]) for r in records
            ),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        files[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records": len(records),
            "source_counts": payload["source_counts"],
        }

    save(
        "validation.json",
        validation,
        "fitting validation; excluded from every nested fit",
    )
    for n in [128, 512, 2048, 8192, 16384, 32768]:
        if n <= len(training):
            save(f"train-{n}.json", training[:n], "nested fitting data")
    if args.max_records < 128:
        save(f"train-{args.max_records}.json", training, "infrastructure pilot only")
    save(
        "train-diagnostic.json",
        training[: min(256, len(training))],
        "fixed in-training diagnostics",
    )
    (args.output / "manifest-gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "files": files,
                "provenance": provenance,
                "elapsed_seconds": time.monotonic() - started,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "records": len(training),
                "exclusions": dict(exclusions),
                "elapsed_seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
