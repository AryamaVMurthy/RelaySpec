"""Audit all fixed research shards and score saved answers with the pinned scorer."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml
from aggregate_results import build_math_scorer

from relayspec.ar_paper_evidence import summarize
from relayspec.paired_accuracy import paired_accuracy_interval


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--wave", type=Path, required=True)
    parser.add_argument("--scoring-repo", type=Path, required=True)
    parser.add_argument("--secondary", action="store_true")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/autoresearch/20260907/confirmation-v1/protocol.json"),
    )
    args = parser.parse_args()
    run, wave = args.run.resolve(), args.wave.resolve()
    repo = Path.cwd()
    protocol_path = repo / args.protocol
    protocol = json.loads(protocol_path.read_text())
    specs = json.loads(wave.read_text())["lanes"]
    if "primary_candidate" in protocol:
        for spec in specs:
            paths = {spec["checkpoint"], *spec["candidates"].values()}
            if paths != set(protocol["checkpoints"]):
                raise ValueError("Candidate set differs from frozen protocol")
            if spec["checkpoint_sha256"] != protocol["checkpoints"]:
                raise ValueError("Checkpoint hashes differ from frozen protocol")
            if protocol["primary_candidate"] not in spec["candidates"]:
                raise ValueError("Frozen primary candidate is absent")
    status = json.loads((run / "wave-result.json").read_text())["lanes"]
    if len(status) != 4 or any(s["status"] != "pass" for s in status):
        raise ValueError("All four shards must pass; no partial confirmation")
    inputs = {str(protocol_path): digest(protocol_path), str(wave): digest(wave)}
    rows, all_ids = [], set()
    for i, spec in enumerate(specs):
        lane = run / f"lane{i}"
        actual = yaml.safe_load((lane / "config.yaml").read_text())
        expected = yaml.safe_load((repo / spec["config"]).read_text())
        manifest_path = repo / expected["benchmark"]["manifest_path"]
        ids = {
            r["problem_id"] for r in json.loads(manifest_path.read_text())["records"]
        }
        methods = {"relay_base", *spec["candidates"]}
        if spec.get("include_native"):
            methods.add(
                "native_target_eagle3"
                if spec["family"] == "eagle3"
                else "native_target_dflash"
            )
        if (
            all_ids & ids
            or len(ids) != 16
            or actual["generation"]
            != {**expected["generation"], "max_new_tokens": 2048}
            or set(actual["benchmark"]["methods"]) != methods
        ):
            raise ValueError(
                "Shard settings or request overlap differ from declaration"
            )
        for field in ["target", "proposer", "source_trunk", "seed"]:
            if actual[field] != expected[field]:
                raise ValueError(f"Frozen model/settings mismatch: {field}")
        all_ids |= ids
        raw_path = lane / "benchmark-rank0.jsonl"
        raw = [json.loads(x) for x in raw_path.read_text().splitlines()]
        keys = {(r["problem_id"], r["method"], r["repetition"]) for r in raw}
        if keys != {(p, m, 0) for p in ids for m in methods} or len(raw) != len(keys):
            raise ValueError("Missing, extra or duplicate request/method")
        for r in raw:
            cp = (
                spec["checkpoint"]
                if r["method"] == "relay_base"
                else spec["candidates"].get(r["method"])
            )
            if cp and r["mapper_checkpoint_sha256"] != spec["checkpoint_sha256"][cp]:
                raise ValueError("Checkpoint identity mismatch")
        rows.extend(raw)
        for p in [
            manifest_path,
            lane / "config.yaml",
            raw_path,
            lane / "completion-gate.json",
        ]:
            inputs[str(p)] = digest(p)
    if len(all_ids) != protocol["records"]:
        raise ValueError("Wrong total request count")
    reserve = (
        repo / "configs/submission/confirmation-gsm8k-20260906/gsm8k-reserve-256.json"
    )
    if digest(reserve) != protocol["reserve_sha256"]:
        raise ValueError("Reserve identity changed")
    frozen_ids = sorted(
        r["problem_id"] for r in json.loads(reserve.read_text())["records"]
    )[protocol.get("reserve_start", 0) : protocol.get("reserve_stop", 64)]
    if all_ids != set(frozen_ids):
        raise ValueError("Confirmation requests differ from frozen selection")
    os.chdir(args.scoring_repo)
    provenance_path = Path("reports/ar-revision-20260905/scorer-provenance.json")
    provenance = json.loads(provenance_path.read_text())
    for name, sha in provenance["sha256"].items():
        if digest(Path(name)) != sha:
            raise ValueError(f"Pinned scorer changed: {name}")
    scorer, name = build_math_scorer(Path("vendor/qwen-math/evaluation"))
    cache = {}
    for r in rows:
        key = (r["completion"], str(r["reference_answer"]), r["benchmark"])
        if key not in cache:
            cache[key] = scorer(*key)
        r.update(cache[key])
    reference = protocol.get(
        "reference", "native_target_dflash" if args.secondary else "relay_base"
    )
    candidate = protocol.get(
        "primary_candidate", "relay_svd1536" if args.secondary else "relay_last2"
    )
    summary = summarize(rows, reference=reference)
    lookup = {(r["problem_id"], r["method"]): r for r in rows}
    ordered = sorted(all_ids)
    accuracy = paired_accuracy_interval(
        [lookup[p, candidate]["correct"] for p in ordered],
        [lookup[p, reference]["correct"] for p in ordered],
    )
    result = dict(
        status="complete",
        secondary=args.secondary,
        summary=summary,
        paired_accuracy=accuracy,
        retention_pass=None
        if args.secondary
        else summary["methods"][candidate]["throughput_ci95"][0] >= 0.95,
        cap_length_counts={
            m: sum(r["method"] == m and r["output_tokens"] == 2048 for r in rows)
            for m in summary["methods"]
        },
        scorer=name,
        scorer_provenance_sha256=digest(provenance_path),
        input_sha256=inputs,
        scope="All64 fixed paired requests. Paired throughput bootstrap and conservative paired accuracy interval. Cap-length counts are not exact truncation counts; no uncapped or1-point quality noninferiority claim. Secondary comparison is descriptive.",
    )
    (run / "confirmation-scored.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    (run / "confirmation-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["input_sha256", "summary"]}
        )
    )


if __name__ == "__main__":
    main()
