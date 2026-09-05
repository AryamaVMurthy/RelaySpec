from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_summarizer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_evalplus.py"
    spec = importlib.util.spec_from_file_location("evalplus_summarizer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_collects_official_base_pass_at_one(tmp_path: Path) -> None:
    export = tmp_path / "evalplus"
    export.mkdir()
    samples = export / "mbpp-relay_f.jsonl"
    samples.write_text("{}\n", encoding="utf-8")
    (export / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "benchmark": "mbpp",
                        "method": "relay_f",
                        "path": str(samples),
                        "samples": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result_path = export / "mbpp-relay_f-sanitized-full.eval_results.json"
    result_path.write_text(
        json.dumps(
            {
                "pass_at_k": {
                    "base": {"pass@1": 0.5},
                    "plus": {"pass@1": 0.5},
                },
                "eval": {
                    "Mbpp/1": [{"base_status": "pass", "plus_status": "pass"}],
                    "Mbpp/2": [{"base_status": "fail", "plus_status": "pass"}],
                },
            }
        ),
        encoding="utf-8",
    )

    summary = load_summarizer().summarize(tmp_path)

    assert summary["benchmarks"]["mbpp"]["relay_f"]["samples"] == 2
    assert summary["benchmarks"]["mbpp"]["relay_f"]["base_pass_at_1"] == 0.5
    assert summary["benchmarks"]["mbpp"]["relay_f"]["base_passed"] == 1
    assert summary["benchmarks"]["mbpp"]["relay_f"]["plus_pass_at_1"] == 0.5


def test_summarize_rejects_incomplete_official_results(tmp_path: Path) -> None:
    export = tmp_path / "evalplus"
    export.mkdir()
    samples = export / "humaneval-native_ar.jsonl"
    samples.write_text("{}\n", encoding="utf-8")
    (export / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "benchmark": "humaneval",
                        "method": "native_ar",
                        "path": str(samples),
                        "samples": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (export / "humaneval-native_ar-sanitized.eval_results.json").write_text(
        json.dumps(
            {
                "pass_at_k": {"base": {"pass@1": 1.0}},
                "eval": {"HumanEval/0": [{"base_status": "pass"}]},
            }
        ),
        encoding="utf-8",
    )

    try:
        load_summarizer().summarize(tmp_path)
    except ValueError as error:
        assert "expected 2" in str(error)
    else:
        raise AssertionError("incomplete EvalPlus result was accepted")


def test_summarize_reports_paired_official_quality_delta(tmp_path: Path) -> None:
    export = tmp_path / "evalplus"
    export.mkdir()
    files = []
    for method in ("optimized_source_reuse", "relay_p"):
        samples = export / f"mbpp-{method}.jsonl"
        samples.write_text("{}\n", encoding="utf-8")
        files.append(
            {
                "benchmark": "mbpp",
                "method": method,
                "path": str(samples),
                "samples": 2,
            }
        )
    (export / "manifest.json").write_text(
        json.dumps({"files": files}), encoding="utf-8"
    )
    statuses = {
        "optimized_source_reuse": [("pass", "pass"), ("fail", "fail")],
        "relay_p": [("pass", "pass"), ("pass", "fail")],
    }
    for method, values in statuses.items():
        base_rate = sum(base == "pass" for base, _ in values) / 2
        plus_rate = sum(base == "pass" and plus == "pass" for base, plus in values) / 2
        (export / f"mbpp-{method}-sanitized-full.eval_results.json").write_text(
            json.dumps(
                {
                    "pass_at_k": {
                        "base": {"pass@1": base_rate},
                        "plus": {"pass@1": plus_rate},
                    },
                    "eval": {
                        f"Mbpp/{index}": [{"base_status": base, "plus_status": plus}]
                        for index, (base, plus) in enumerate(values)
                    },
                }
            ),
            encoding="utf-8",
        )

    summary = load_summarizer().summarize(tmp_path)

    paired = summary["paired_quality"]["mbpp"]["relay_p"]
    assert summary["reference_method"] == "optimized_source_reuse"
    assert paired["base"]["accuracy_delta"] == 0.5
    assert paired["plus"]["accuracy_delta"] == 0.0
    assert paired["base"]["candidate_only_passes"] == 1
