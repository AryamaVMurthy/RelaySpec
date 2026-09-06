"""Audit equal-data adaptation timing, identity controls, and saved decoding."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from relayspec.ar_paper_evidence import read_rows, summarize
from relayspec.mapper_campaign import campaign_references
from relayspec.quality_scoring import verify_saved_scores


def family_evidence_fields(family):
    if family == "dflash":
        return (
            "sha256",
            "drafter_checkpoint_sha256",
            ("accepted_draft_lengths", "proposal_lengths"),
        )
    if family == "eagle3":
        return (
            "checkpoint_sha256",
            "drafter_update_sha256",
            ("acceptance_lengths", "target_calls", "draft_calls"),
        )
    raise ValueError("unsupported adaptation evidence family")


def check_interface_proof(fit, family):
    if family == "dflash":
        if fit["direct_fusion_equivalence"]["status"] != "pass":
            raise ValueError(
                "DFlash calibration lacks its direct-fusion placement check"
            )
    elif family == "eagle3":
        checks = fit.get("eagle_initial_cache_equivalence", [])
        if len(checks) != 4 or any(
            c.get("status") != "pass" or c.get("logits_bit_identical") is not True
            for c in checks
        ):
            raise ValueError("EAGLE adaptation lacks shifted-prefix cache equivalence")
    else:
        raise ValueError("unsupported adaptation evidence family")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--scoring-repo", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    settings = config["adaptation_pilot"]
    family = config["proposer"]["family"]
    mapper_key, drafter_key, acceptance_fields = family_evidence_fields(family)
    gate_path = args.run / "adaptation-pilot-gate.json"
    gate = json.loads(gate_path.read_text())
    if gate["status"] != "pass" or gate["config_sha256"] != digest(args.config):
        raise ValueError("adaptation calibration has not passed its declared pilot")
    inputs = {str(args.config): digest(args.config), str(gate_path): digest(gate_path)}
    if family == "eagle3" and not args.ledger:
        raise ValueError(
            "EAGLE adaptation audit requires its immutable submission ledger"
        )
    if args.ledger:
        ledger = json.loads(args.ledger.read_text())
        matches = [j for j in ledger["jobs"] if Path(j["local"]).name == args.run.name]
        source = args.run / "source-commit.txt"
        if len(matches) != 1 or source.read_text().strip() != ledger["source_commit"]:
            raise ValueError("adaptation run differs from its submission source")
        inputs.update({str(p): digest(p) for p in [source, args.ledger]})
    fits = []
    for rank in range(4):
        folder = args.run / "fitting" / f"rank{rank}"
        fit_path = folder / "adaptation-fit-gate.json"
        history_path = folder / "training.jsonl"
        fit = json.loads(fit_path.read_text())
        history = [json.loads(line) for line in history_path.read_text().splitlines()]
        if (
            fit != gate["fitting"][rank]
            or fit["status"] != "pass"
            or fit["updates"] != settings["pilot_updates"]
            or fit["distinct_examples"] != settings["pilot_distinct_examples"]
            or len(set(fit["ordered_record_files"])) != fit["distinct_examples"]
            or [h["step"] for h in history] != list(range(1, fit["updates"] + 1))
            or history[-1]["presentations"] != fit["presentations"]
            or not fit["inherited_weights_unchanged"]
        ):
            raise ValueError("missing equal-data fit, history, or identity evidence")
        check_interface_proof(fit, family)
        if rank and fit["ordered_record_files"] != fits[0]["ordered_record_files"]:
            raise ValueError("adaptation workers used different ordered records")
        inputs.update(
            {str(fit_path): digest(fit_path), str(history_path): digest(history_path)}
        )
        fits.append(fit)
    if fits[2]["export"]["checkpoint_sha256"] != fits[3]["export"]["checkpoint_sha256"]:
        raise ValueError("duplicate adapted weights differ")
    evaluation = args.run / "evaluation"
    completion = json.loads((evaluation / "completion-gate.json").read_text())
    analysis = json.loads((evaluation / "analysis.json").read_text())
    manifest = json.loads((evaluation / "mapper-campaign.json").read_text())
    for path in [
        evaluation / name
        for name in [
            "completion-gate.json",
            "analysis.json",
            "mapper-campaign.json",
            "config.yaml",
        ]
    ]:
        inputs[str(path)] = digest(path)
    if completion["status"] != "pass" or analysis["status"] != "complete":
        raise ValueError("adaptation decoding is incomplete")
    for name, sha in analysis["raw_sha256"].items():
        path = evaluation / name
        if digest(path) != sha:
            raise ValueError("raw adaptation decoding changed")
        inputs[str(path)] = sha
    rows = read_rows(evaluation)
    expected_methods = [
        *campaign_references(family, config["benchmark"]["methods"]),
        "relay_initial",
        "relay_connector_ce",
        "relay_lora8",
        "relay_lora32",
        "relay_lora32_duplicate",
        "relay_lora_zero",
    ]
    actual_config = yaml.safe_load((evaluation / "config.yaml").read_text())
    if (
        set(r["method"] for r in rows) != set(expected_methods)
        or actual_config["benchmark"]["methods"] != expected_methods
        or any(
            actual_config[key] != config[key]
            for key in ["target", "proposer", "source_trunk"]
        )
    ):
        raise ValueError(
            "adaptation decoding omits a method or changes model configuration"
        )
    for row in rows:
        variant = manifest["variants"].get(row["method"])
        if variant and row.get("mapper_checkpoint_sha256") != variant[mapper_key]:
            raise ValueError("decoding used another mapper")
        if (
            variant
            and "drafter_update" in variant
            and row.get(drafter_key) != variant["drafter_update"]["sha256"]
        ):
            raise ValueError("decoding used another drafter update")
    for method, rank in [
        ("relay_lora8", 1),
        ("relay_lora32", 2),
        ("relay_lora32_duplicate", 3),
    ]:
        if (
            manifest["variants"][method]["drafter_update"]["sha256"]
            != fits[rank]["export"]["checkpoint_sha256"]
        ):
            raise ValueError("decoded drafter differs from its verified export")
    for left, right in [
        ("relay_initial", "relay_lora_zero"),
        ("relay_lora32", "relay_lora32_duplicate"),
    ]:
        pairs = [
            {(r["problem_id"], r["repetition"]): r for r in rows if r["method"] == name}
            for name in (left, right)
        ]
        if pairs[0].keys() != pairs[1].keys() or not pairs[0]:
            raise ValueError("identity control requests differ")
        for key, row in pairs[0].items():
            if any(
                row[field] != pairs[1][key][field]
                for field in ("output_hash", "output_tokens", *acceptance_fields)
            ):
                raise ValueError("identity control decoding differs")
    paired = summarize(rows, reference="relay_initial")
    if paired["requests"] != 8 or len(rows) != completion["records"]:
        raise ValueError("pilot is not the declared eight-request comparison")
    if family == "eagle3":
        scoring_repo = args.scoring_repo or args.run.resolve().parents[3]
        verify_saved_scores(evaluation, scoring_repo)
        for path in [
            evaluation / "math-scored.jsonl",
            scoring_repo / "reports/ar-revision-20260905/scorer-provenance.json",
            scoring_repo / "scripts/analyze_controlled_run.py",
        ]:
            inputs[str(path)] = digest(path)
    result = {
        "status": "complete",
        "input_sha256": inputs,
        "fitting": fits,
        "against_initial_mapper": paired,
        "scope": "Equal-data 512-record one-pass calibration, not matched-compute "
        "or final task-quality evidence. Fitting-loop time excludes model/teacher "
        "preparation, initial mapper fitting, shared feature-cache construction, "
        "checkpoint export and decoding. Preparation and total worker times are "
        "reported separately. Eight exposed requests, 128-token cap, greedy decoding. "
        "Direct fusion equality covers four teacher-forced records. Raw trainable "
        "state is serialized, but continuation equality still requires validation.",
    }
    if family == "eagle3":
        result.update(
            family=family,
            stage="compatibility_pilot"
            if settings["pilot_distinct_examples"] == 64
            else "equal_data_calibration",
            scope=f"EAGLE-3 one-step shifted teacher-forced adaptation on {settings['pilot_distinct_examples']} existing cached records and {settings['pilot_updates']} batch-four updates. Four-record initial-cache/logit equivalence, frozen inherited weights, merged reload, duplicate-seed fitting and full-output decoding identity. Not multi-step TTT, matched-time comparison, final quality or a direct-fusion deployment claim. Initial mapper fitting, feature-cache construction, teacher preparation, optimizer updates, export and decoding are separate costs. Eight exposed requests at128 tokens.",
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {
                    k: values[k]
                    for k in [
                        "tokens_per_second",
                        "throughput_ratio",
                        "throughput_ci95",
                    ]
                }
                for name, values in paired["methods"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
