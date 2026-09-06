"""Audit exact composition cache membership and its compatibility-pilot evidence."""

import argparse
import copy
import json
import random
from pathlib import Path

import yaml

from relayspec.ar_paper_evidence import read_rows
from relayspec.cache_data_inputs import audit_manifest_entries, checked_data_inputs
from relayspec.cached_fit_evidence import audit_fit_artifacts, digest


def audit_pilot_fits(run, cache_sha, inputs):
    trials_path = run / "cached-trials.json"
    trials = json.loads(trials_path.read_text())["trials"]
    if [t["name"] for t in trials] != ["dense", "factor512", "mlp512", "mlp512-l2"]:
        raise ValueError("composition pilot requires its four declared fit paths")
    results = {}
    for trial, architecture, l2 in zip(
        trials, ["dense", "factorized", "mlp", "mlp"], [0, 0, 0, 1e-6], strict=True
    ):
        if (
            trial["architecture"] != architecture
            or trial["steps"] != 16
            or trial["distinct_examples"] != 64
            or trial["seed"] != 1729
            or trial["learning_rate"] != 6e-4
            or trial["l2_weight"] != l2
            or trial["weight_decay"] != 0
            or trial["validation_records"] != 16
            or trial["checkpoint_steps"] != [8, 16]
            or (architecture != "dense" and trial["width"] != 512)
            or trial.get("report_domain_diagnostics") is not True
            or set(trial["validation_required_domains"])
            != {"math", "general_instruction"}
        ):
            raise ValueError(
                "composition compatibility fit differs from pilot protocol"
            )
        folder = run / "fitting" / trial["name"]
        result = audit_fit_artifacts(folder, trial, cache_sha256=cache_sha)
        equivalence_path = folder / "equivalence-gate.json"
        equivalence = json.loads(equivalence_path.read_text())
        if (
            equivalence.get("status") != "pass"
            or not 0 <= equivalence["relative_gradient_error"] <= 0.02
            or not 0 <= equivalence["relative_loss_error"] <= 0.01
        ):
            raise ValueError("composition pilot failed its padded-batch gradient check")
        inputs[str(equivalence_path)] = digest(equivalence_path)
        for name, sha in result["source_sha256"].items():
            inputs[str(folder / name)] = sha
        results[trial["name"]] = result
    inputs[str(trials_path)] = digest(trials_path)
    evaluation = run / "evaluation"
    config_path = run / "campaign-config.yaml"
    config = yaml.safe_load(config_path.read_text())
    gate_path = evaluation / "campaign-gate.json"
    gate = json.loads(gate_path.read_text())
    complete_path = evaluation / "completion-gate.json"
    complete = json.loads(complete_path.read_text())
    if (
        gate.get("status") != "pass"
        or gate["duplicate_map_equivalence"]["status"] != "pass"
        or complete.get("status") != "pass"
        or config["benchmark"]["max_prompts"] != 8
        or config["generation"]["max_new_tokens"] != 128
    ):
        raise ValueError("composition pilot did not pass its bounded decoding check")
    manifest_path = Path(config["benchmark"]["manifest_path"])
    selected = [
        r
        for r in json.loads(manifest_path.read_text())["records"]
        if r["benchmark"] in config["benchmark"]["benchmarks"]
    ]
    random.Random(config["seed"]).shuffle(selected)
    selected = selected[:8]
    expected = {
        (r["problem_id"], 0, method)
        for r in selected
        for method in config["benchmark"]["methods"]
    }
    rows = read_rows(evaluation)
    if (
        len(rows) != len(expected)
        or {(r["problem_id"], r["repetition"], r["method"]) for r in rows} != expected
        or complete["records"] != len(rows)
        or complete["requests"] != 8
    ):
        raise ValueError("composition decoding lacks the exact common requests")
    pairs = [
        {(r["problem_id"], r["repetition"]): r for r in rows if r["method"] == name}
        for name in ["relay_dense_a", "relay_dense_b"]
    ]
    if len(pairs[0]) != 8 or pairs[0].keys() != pairs[1].keys():
        raise ValueError("composition duplicate-map control is incomplete")
    fields = [
        "output_hash",
        "output_tokens",
        "mapper_checkpoint_sha256",
        "accepted_draft_lengths",
        "proposal_lengths",
    ]
    if any(
        pairs[0][key][field] != pairs[1][key][field]
        for key in pairs[0]
        for field in fields
    ):
        raise ValueError("composition duplicate maps differ in actual raw decoding")
    for path in [
        config_path,
        manifest_path,
        gate_path,
        complete_path,
        *sorted(evaluation.glob("benchmark-rank*.jsonl")),
    ]:
        inputs[str(path)] = digest(path)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pilot", "full"], required=True)
    parser.add_argument("--runs", type=Path, nargs=2, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/composition/math-dolly-v2")
    )
    parser.add_argument("--pilot-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = Path(
        "configs/submission/scaling/composition-small-v1/protocol.json"
    )
    protocol, ledger = [json.loads(p.read_text()) for p in [protocol_path, args.ledger]]
    training_path = Path(protocol["training_config"])
    training = yaml.safe_load(training_path.read_text())
    inputs = {
        str(p): digest(p)
        for p in [
            protocol_path,
            args.ledger,
            training_path,
            args.data_root / "manifest-gate.json",
        ]
    }
    if protocol["manifest_gate_sha256"] != digest(
        args.data_root / "manifest-gate.json"
    ):
        raise ValueError("composition manifest declaration changed")
    prior = None
    if args.stage == "full":
        if not args.pilot_result:
            raise ValueError(
                "full composition caches require both audited compatibility pilots"
            )
        prior = json.loads(args.pilot_result.read_text())
        if (
            prior.get("status") != "complete"
            or prior["stage"] != "pilot"
            or prior["protocol_sha256"] != digest(protocol_path)
        ):
            raise ValueError("composition pilots did not pass this exact protocol")
        for name, sha in prior["input_sha256"].items():
            if digest(Path(name)) != sha:
                raise ValueError("composition pilot evidence changed")
            inputs[name] = sha
        inputs[str(args.pilot_result)] = digest(args.pilot_result)
    counts = (
        {"train": 64, "validation": 16}
        if args.stage == "pilot"
        else {"train": 2048, "validation": 2048}
    )
    records = {}
    for run in args.runs:
        run = run.resolve()
        job = next(j for j in ledger["jobs"] if Path(j["local"]).name == run.name)
        arm = job["arm"]
        if (
            arm not in {"math", "mixed"}
            or arm in records
            or (run / "source-commit.txt").read_text().strip()
            != ledger["source_commit"]
        ):
            raise ValueError("composition cache source/arm is undeclared or duplicated")
        filename = protocol["arms"][arm]["training_manifest"]
        expected_inputs = checked_data_inputs(
            args.data_root,
            filename,
            train_records=counts["train"],
            validation_records=counts["validation"],
            report_domains=True,
        )
        gate_path = run / (
            "pilot-gate.json" if args.stage == "pilot" else "extraction-complete.json"
        )
        gate = json.loads(gate_path.read_text())
        cache_gate_path, index_path = (
            run / "feature-cache-gate.json",
            run / "cache-index.json",
        )
        cache_gate, index = [
            json.loads(p.read_text()) for p in [cache_gate_path, index_path]
        ]
        config_path = run / "cache-config.yaml"
        config = yaml.safe_load(config_path.read_text())
        expected_config = copy.deepcopy(training)
        expected_config["relay_training"].update(
            manifest_path=config["relay_training"]["manifest_path"],
            validation_manifest_path=config["relay_training"][
                "validation_manifest_path"
            ],
            feature_cache={
                "train_records": counts["train"],
                "validation_records": counts["validation"],
            },
        )
        if (
            gate.get("status") != "pass"
            or gate["data_inputs"] != expected_inputs
            or cache_gate.get("status") != "pass"
            or cache_gate["counts"] != counts
            or digest(index_path) != cache_gate["cache_index_sha256"]
            or cache_gate["metadata"] != index["metadata"]
            or config != expected_config
            or {
                **index["metadata"]["config"],
                "relay_training": index["metadata"]["relay_training"],
            }
            != config
            or index["metadata"]["config_sha256"] != digest(config_path)
        ):
            raise ValueError(
                "composition cache differs from declared data/config or copied index"
            )
        groups = {}
        for split, name, field in [
            ("train", filename, "manifest_path"),
            ("validation", "validation.json", "validation_manifest_path"),
        ]:
            path = args.data_root / name
            remote_path = config["relay_training"][field]
            if Path(remote_path).name != name or index["metadata"][
                "manifest_sha256"
            ].get(remote_path) != digest(path):
                raise ValueError("cache extraction used a different manifest")
            groups[split] = json.loads(path.read_text())["records"][: counts[split]]
            inputs[str(path)] = digest(path)
        coverage = audit_manifest_entries(index, groups)
        if (
            cache_gate["total_bytes"] != index["total_bytes"]
            or cache_gate["total_tokens"] != index["total_tokens"]
        ):
            raise ValueError("composition gate totals differ from index entries")
        result = {
            "job": job["id"],
            "cache_root": cache_gate["cache_root"],
            "cache_index_sha256": digest(index_path),
            "gate_sha256": digest(gate_path),
            "coverage": coverage,
        }
        if args.stage == "pilot":
            if (
                gate["training_config_sha256"] != digest(training_path)
                or gate["feature_cache_index_sha256"] != digest(index_path)
                or gate["models"] != {k: training[k] for k in ["target", "proposer"]}
            ):
                raise ValueError(
                    "compatibility pilot did not use declared model configuration"
                )
            result["fits"] = audit_pilot_fits(run, digest(index_path), inputs)
        elif (
            gate["counts"] != counts
            or gate["cache_index_sha256"] != digest(index_path)
            or gate["pilot_gate_sha256"] != prior["records"][arm]["gate_sha256"]
        ):
            raise ValueError("proper extraction is not bound to its matching pilot")
        else:
            result["extraction_wall_seconds"] = gate[
                "wall_seconds_including_model_load"
            ]
        records[arm] = result
        for path in [
            gate_path,
            cache_gate_path,
            index_path,
            config_path,
            run / "source-commit.txt",
        ]:
            inputs[str(path)] = digest(path)
    if set(records) != {"math", "mixed"}:
        raise ValueError("composition requires both matched-count arms")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "complete",
                "stage": args.stage,
                "protocol_sha256": digest(protocol_path),
                "input_sha256": inputs,
                "records": records,
                "scope": "Exact copied cache-index membership and domain/token accounting, source/config and pilot links. Actual tensor bytes are separately hash-checked by GPU fit consumers. Compatibility fits/decoding establish infrastructure only; proper composition fits, downstream decoding and quality remain separate.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Audited both composition {args.stage} caches")


if __name__ == "__main__":
    main()
