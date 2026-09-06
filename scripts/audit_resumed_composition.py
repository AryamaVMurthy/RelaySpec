"""Audit declared composition fits, including exact-cache pilot equivalence."""

import argparse
import json
from pathlib import Path
from relayspec.cached_fit_evidence import audit_fit_artifacts, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--arm", choices=["math", "mixed"], required=True)
    parser.add_argument("--stage", choices=["pilot", "fits"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = (
        Path("configs/submission/scaling/composition-small-v1")
        / f"{args.arm}-{args.stage}.json"
    )
    trials = json.loads(source.read_text())["trials"]
    cache = json.loads(
        Path(
            "reports/mapper-scaling-20260905/composition-cache-results.json"
        ).read_text()
    )["records"][args.arm]["cache_index_sha256"]
    gate = json.loads((args.run / "batch-gate.json").read_text())
    assert gate["status"] == "pass" and gate["mode"] == (
        "pilot" if args.stage == "pilot" else "fit"
    )
    assert gate["feature_cache_index_sha256"] == cache and gate[
        "trials_sha256"
    ] == digest(source)
    assert gate["trials"] == [t["name"] for t in trials]
    results = {}
    for t in trials:
        folder = args.run / "fitting" / t["name"]
        results[t["name"]] = audit_fit_artifacts(folder, t, cache_sha256=cache)
        assert {
            Path(p).name
            for p in gate["checkpoint_sha256"]
            if Path(p).parent.name == t["name"]
        } == {f"step-{s:06d}.pt" for s in t["checkpoint_steps"]}
        if args.stage == "pilot":
            eq = json.loads((folder / "equivalence-gate.json").read_text())
            assert (
                eq["status"] == "pass"
                and eq["relative_gradient_error"] <= eq["gradient_tolerance"]
                and eq["relative_loss_error"] <= eq["loss_tolerance"]
            )
    if args.stage == "pilot":
        eg = json.loads((args.run / "evaluation/campaign-gate.json").read_text())
        cg = json.loads((args.run / "evaluation/completion-gate.json").read_text())
        assert (
            eg["status"] == "pass"
            and cg["status"] == "pass"
            and cg["records"] == 64
            and cg["requests"] == 8
        )
        rows = [
            json.loads(l)
            for p in (args.run / "evaluation").glob("benchmark-rank*.jsonl")
            for l in p.read_text().splitlines()
        ]
        pairs = {}
        for r in rows:
            pairs.setdefault(r["problem_id"], {})[r["method"]] = r
        assert len(rows) == 64 and len(pairs) == 8
        for group in pairs.values():
            assert set(group) == set(eg["methods"])
            for key in eg["duplicate_map_equivalence"]["checked_fields"]:
                assert group["relay_0"][key] == group["relay_duplicate"][key]
    output = {
        "status": "complete",
        "arm": args.arm,
        "stage": args.stage,
        "cache_sha256": cache,
        "results": results,
        "input_sha256": {
            str(p): digest(p)
            for p in [
                source,
                *sorted(args.run.rglob("*.json")),
                *sorted(args.run.rglob("*.jsonl")),
            ]
        },
        "scope": "Reconstructed training exposure, parameter counts, domain validation and cost. Pilot also rechecks numerical equivalence and paired duplicate-map outputs. Fitting evidence alone is not downstream quality evidence.",
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(args.arm, args.stage, len(results), "audited")


if __name__ == "__main__":
    main()
