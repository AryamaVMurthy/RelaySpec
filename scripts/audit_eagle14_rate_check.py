"""Audit the four declared EAGLE14 optimization checks without selecting a horizon."""

import argparse
import json
from pathlib import Path

from relayspec.ar_paper_evidence import read_rows
from relayspec.cached_fit_evidence import audit_fit_artifacts, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pilot", "fit"], required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--pilot-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_root = Path("configs/submission/scaling/target14b-eagle-rate-check")
    protocol_path = config_root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    ledger = json.loads(args.ledger.read_text())
    if len(ledger["jobs"]) != 1:
        raise ValueError("rate check requires exactly one four-worker batch")
    job = ledger["jobs"][0]
    folder = args.raw_root / job["local"]
    inputs = {str(p): digest(p) for p in [protocol_path, args.ledger]}
    for name, sha in protocol["input_sha256"].items():
        if digest(Path(name)) != sha:
            raise ValueError("rate-check selection evidence changed")
        inputs[name] = sha
    original = json.loads(Path(next(iter(protocol["input_sha256"]))).read_text())
    cache_sha = original["feature_cache_index_sha256"]
    template = Path(
        "configs/submission/scaling/target14b-small-v1/campaign-eagle3-pilot.yaml"
    )
    trials_path = config_root / f"{args.stage}.json"
    trials = json.loads(trials_path.read_text())["trials"]
    expected = []
    for declared in protocol["trials"]:
        trial = dict(declared)
        if args.stage == "pilot":
            trial.update(
                name="pilot-" + trial["name"],
                steps=16,
                checkpoint_steps=[16],
                budget_panels={},
            )
        expected.append(trial)
    gate_path = folder / "batch-gate.json"
    gate = json.loads(gate_path.read_text())
    if (
        len(trials) != 4
        or trials != expected
        or gate.get("status") != "pass"
        or gate["mode"] != args.stage
        or gate["trials"] != [t["name"] for t in trials]
        or gate["trials_sha256"] != digest(trials_path)
        or digest(folder / "trials.json") != digest(trials_path)
        or gate["campaign_config_sha256"] != digest(template)
        or gate["feature_cache_index_sha256"] != cache_sha
        or (folder / "source-commit.txt").read_text().strip() != ledger["source_commit"]
    ):
        raise ValueError("rate batch differs from its declared source/trials/cache")
    for p in [
        gate_path,
        trials_path,
        template,
        folder / "source-commit.txt",
        folder / "trials.json",
    ]:
        inputs[str(p)] = digest(p)
    if args.stage == "fit":
        if not args.pilot_result:
            raise ValueError("proper rate checks require the audited matching pilot")
        pilot = json.loads(args.pilot_result.read_text())
        if (
            pilot["status"] != "complete"
            or pilot["stage"] != "pilot"
            or pilot["protocol_sha256"] != digest(protocol_path)
            or gate["pilot_gate_sha256"] != pilot["batch_gate_sha256"]
        ):
            raise ValueError("proper rate checks used a different pilot")
        for name, sha in pilot["input_sha256"].items():
            if digest(Path(name)) != sha:
                raise ValueError("pilot evidence changed")
            inputs[name] = sha
        inputs[str(args.pilot_result)] = digest(args.pilot_result)
    results = {}
    for trial in trials:
        fit_dir = folder / "fitting" / trial["name"]
        record = audit_fit_artifacts(fit_dir, trial, cache_sha256=cache_sha)
        if args.stage == "pilot":
            equivalence_path = fit_dir / "equivalence-gate.json"
            equivalence = json.loads(equivalence_path.read_text())
            if (
                equivalence.get("status") != "pass"
                or not 0 <= equivalence["relative_gradient_error"] <= 0.02
                or not 0 <= equivalence["relative_loss_error"] <= 0.01
            ):
                raise ValueError("rate pilot failed padded-batch gradient equivalence")
            inputs[str(equivalence_path)] = digest(equivalence_path)
        for name, sha in record["source_sha256"].items():
            inputs[str(fit_dir / name)] = sha
        checkpoints = {}
        for step in trial["checkpoint_steps"]:
            matches = [
                (name, sha)
                for name, sha in gate["checkpoint_sha256"].items()
                if Path(name).parts[-2:] == (trial["name"], f"step-{step:06d}.pt")
            ]
            if len(matches) != 1 or len(matches[0][1]) != 64:
                raise ValueError("rate check lacks a unique declared checkpoint")
            checkpoints.update(matches)
        results[trial["name"]] = {
            **record,
            "checkpoint_sha256": checkpoints,
            "raw_directory": str(fit_dir),
            "batch_gate_path": str(gate_path),
        }
    if args.stage == "pilot":
        evaluation = folder / "evaluation"
        campaign_path = evaluation / "campaign-gate.json"
        campaign = json.loads(campaign_path.read_text())
        if (
            campaign["status"] != "pass"
            or campaign["duplicate_map_equivalence"]["status"] != "pass"
        ):
            raise ValueError("rate pilot decoding gate did not pass")
        rows = read_rows(evaluation)
        completion_path = evaluation / "completion-gate.json"
        completion = json.loads(completion_path.read_text())
        methods = {
            "native_ar",
            "native_target_eagle3",
            "source_reuse_eagle3",
            "relay_0",
            "relay_1",
            "relay_2",
            "relay_3",
            "relay_duplicate",
        }
        requests = {(r["problem_id"], r["repetition"]) for r in rows}
        if (
            completion["status"] != "pass"
            or completion["requests"] != 8
            or completion["records"] != 64
            or len(rows) != 64
            or len(requests) != 8
            or {(r["problem_id"], r["repetition"], r["method"]) for r in rows}
            != {(pid, rep, method) for pid, rep in requests for method in methods}
        ):
            raise ValueError("rate pilot lacks complete paired request/method coverage")
        inputs[str(completion_path)] = digest(completion_path)
        pairs = [
            {
                (r["problem_id"], r["repetition"]): r
                for r in rows
                if r["method"] == method
            }
            for method in ["relay_0", "relay_duplicate"]
        ]
        if len(pairs[0]) != 8 or pairs[0].keys() != pairs[1].keys():
            raise ValueError("rate pilot lacks eight duplicate-control pairs")
        fields = [
            "output_hash",
            "output_tokens",
            "mapper_checkpoint_sha256",
            "acceptance_lengths",
            "target_calls",
            "draft_calls",
        ]
        if any(
            pairs[0][key][field] != pairs[1][key][field]
            for key in pairs[0]
            for field in fields
        ):
            raise ValueError("rate pilot duplicate decoding differs")
        for p in [campaign_path, *evaluation.glob("benchmark-rank*.jsonl")]:
            inputs[str(p)] = digest(p)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "complete",
                "stage": args.stage,
                "protocol_sha256": digest(protocol_path),
                "batch_gate_sha256": digest(gate_path),
                "feature_cache_index_sha256": cache_sha,
                "input_sha256": inputs,
                "results": results,
                "fit_wall_seconds": gate["fit_wall_seconds"],
                "scope": "Four targeted lower-rate checks, one fitting seed, fixed endpoint. Local audit reconstructs fitting logs and recorded GPU checkpoint hashes. No remote tensor rehash or full-answer quality claim.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Audited four {args.stage} EAGLE14 rate checks")


if __name__ == "__main__":
    main()
