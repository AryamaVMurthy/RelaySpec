"""Bind all 14B capacity endpoints to their raw paired development decoding."""

import argparse
import json
import random
from pathlib import Path

import yaml

from relayspec.ar_paper_evidence import load_scored, summarize
from relayspec.cached_fit_evidence import digest
from relayspec.mapper_campaign import campaign_references
from relayspec.quality_scoring import verify_saved_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["dflash", "eagle3"], required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--scoring-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path("configs/submission/scaling/target14b-small-v1")
    config_path = root / f"campaign-{args.family}-capacity.yaml"
    provenance_path = config_path.with_suffix(".provenance.json")
    fits_path = Path(
        f"reports/mapper-scaling-20260905/target14b-{args.family}-fit-results.json"
    )
    config = yaml.safe_load(config_path.read_text())
    provenance, fits, ledger = [
        json.loads(p.read_text()) for p in [provenance_path, fits_path, args.ledger]
    ]
    run = args.run.resolve()
    job = next(j for j in ledger["jobs"] if Path(j["local"]).name == run.name)
    inputs = {
        str(p): digest(p)
        for p in [config_path, provenance_path, fits_path, args.ledger]
    }
    for name, sha in fits["input_sha256"].items():
        if digest(Path(name)) != sha:
            raise ValueError("audited fitting evidence changed")
        inputs[name] = sha
    if (
        fits.get("status") != "complete"
        or fits["family"] != args.family
        or provenance["fit_registry_sha256"] != digest(fits_path)
        or provenance["config_sha256"] != digest(config_path)
        or provenance["matrix_sha256"] != fits["matrix_sha256"]
        or provenance["feature_cache_index_sha256"]
        != fits["feature_cache_index_sha256"]
        or provenance["primary_cells"] != 14
        or provenance["dense_seed_controls"] != 2
        or yaml.safe_load((run / "config.yaml").read_text()) != config
        or (run / "source-commit.txt").read_text().strip() != ledger["source_commit"]
        or job["family"] != args.family
        or config["target"] != fits["target"]
        or config["proposer"]["family"] != args.family
    ):
        raise ValueError(
            "capacity campaign differs from its complete fitting declaration"
        )
    variants = provenance["variants"]
    if {v["trial"]["name"] for v in variants.values()} != set(fits["results"]) or len(
        variants
    ) != 16:
        raise ValueError("capacity campaign omits or duplicates fitting endpoints")
    for method, declared in variants.items():
        trial = declared["trial"]
        fit = fits["results"][trial["name"]]
        checkpoint = config["relay_probe"]["variants"][method]
        if (
            trial != fit["trial"]
            or Path(checkpoint).name != "step-008192.pt"
            or fit["checkpoint_sha256"].get(checkpoint) != declared["checkpoint_sha256"]
            or digest(Path(fit["batch_gate_path"])) != declared["fit_gate_sha256"]
        ):
            raise ValueError("capacity mapper differs from its fixed fitting endpoint")
    methods = config["benchmark"]["methods"]
    references = campaign_references(args.family, methods)
    if (
        methods != [*references, *variants]
        or config["generation"]["max_new_tokens"] != 256
    ):
        raise ValueError("capacity references, methods or output cap changed")
    manifest_path = Path(config["benchmark"]["manifest_path"])
    selected = [
        r
        for r in json.loads(manifest_path.read_text())["records"]
        if r["benchmark"] == "math500"
    ]
    random.Random(config["seed"]).shuffle(selected)
    selected = selected[:16]
    if (
        config["benchmark"]["benchmarks"] != ["math500"]
        or config["benchmark"]["max_prompts"] != 16
        or config["relay_probe"]["repetitions"] != 1
    ):
        raise ValueError("capacity request protocol changed")
    answers = {r["problem_id"]: str(r["answer"]) for r in selected}
    expected = {(pid, 0, method) for pid in answers for method in methods}
    rows = load_scored(run)
    if (
        len(answers) != 16
        or len(rows) != len(expected)
        or {(r["problem_id"], r["repetition"], r["method"]) for r in rows} != expected
    ):
        raise ValueError("capacity rows lack exact declared request/method coverage")
    for row in rows:
        variant = variants.get(row["method"])
        if (
            row["benchmark"] != "math500"
            or not 0 < row["output_tokens"] <= 256
            or str(row["reference_answer"]) != answers[row["problem_id"]]
            or not isinstance(row["correct"], bool)
            or (
                variant
                and row.get("mapper_checkpoint_sha256") != variant["checkpoint_sha256"]
            )
        ):
            raise ValueError(
                "capacity row has the wrong checkpoint, answer or output cap"
            )
    gate = json.loads((run / "completion-gate.json").read_text())
    campaign = json.loads((run / "campaign-gate.json").read_text())
    if (
        gate.get("status") != "pass"
        or gate["records"] != len(expected)
        or gate["requests"] != 16
        or campaign.get("status") != "pass"
        or campaign["methods"] != methods
    ):
        raise ValueError("capacity run failed completeness")
    verify_saved_scores(run, args.scoring_repo)
    for p in [
        manifest_path,
        run / "config.yaml",
        run / "source-commit.txt",
        run / "completion-gate.json",
        run / "campaign-gate.json",
        run / "math-scored.jsonl",
        run / "analysis.json",
        *sorted(run.glob("benchmark-rank*.jsonl")),
    ]:
        inputs[str(p)] = digest(p)
    for relative in [
        "reports/ar-revision-20260905/scorer-provenance.json",
        "scripts/analyze_controlled_run.py",
        "src/relayspec/ar_paper_evidence.py",
    ]:
        p = args.scoring_repo / relative
        inputs[str(p)] = digest(p)
    dense = next(
        name
        for name, v in variants.items()
        if v["trial"]["architecture"] == "dense"
        and v["trial"]["distinct_examples"] == 2048
        and v["trial"]["seed"] == 1729
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "complete",
                "family": args.family,
                "job": job["id"],
                "requests": 16,
                "records": len(rows),
                "output_cap": 256,
                "input_sha256": inputs,
                "comparisons": {
                    ref: summarize(rows, reference=ref) for ref in [*references, dense]
                },
                "variants": variants,
                "scope": "All 16 fixed fitting endpoints on common exposed development requests. Truncated-answer scores are diagnostics, not full-answer quality or noninferiority evidence. Request bootstrap intervals do not quantify training-seed uncertainty or correct for model selection. All candidate maps are resident together; memory is not isolated deployment memory.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Audited {len(rows)} paired {args.family}14 capacity rows")


if __name__ == "__main__":
    main()
