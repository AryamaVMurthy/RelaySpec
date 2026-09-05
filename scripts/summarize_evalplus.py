from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from relayspec.metrics import paired_binary_delta_summary


def summarize(output_dir: Path) -> dict[str, Any]:
    export_dir = output_dir / "evalplus"
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "scorer": "EvalPlus base and plus tests",
        "benchmarks": {},
    }
    task_statuses: dict[str, dict[str, dict[str, dict[str, bool]]]] = {}
    revision_path = export_dir / "evalplus-revision.txt"
    if revision_path.exists():
        summary["evalplus_revision"] = revision_path.read_text(encoding="utf-8").strip()
    for entry in manifest["files"]:
        benchmark = str(entry["benchmark"])
        method = str(entry["method"])
        sample_path = Path(str(entry["path"]))
        default_result_path = export_dir / (
            sample_path.stem + "-sanitized.eval_results.json"
        )
        full_result_path = export_dir / (
            sample_path.stem + "-sanitized-full.eval_results.json"
        )
        result_path = (
            full_result_path if full_result_path.exists() else default_result_path
        )
        if not result_path.exists():
            raise FileNotFoundError(f"missing official EvalPlus result: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        task_results = result.get("eval", {})
        expected = int(entry["samples"])
        if len(task_results) != expected:
            raise ValueError(
                f"{benchmark}/{method}: expected {expected} evaluated tasks, "
                f"found {len(task_results)}"
            )
        passed = 0
        for task_id, candidates in task_results.items():
            if len(candidates) != 1:
                raise ValueError(
                    f"{benchmark}/{method}/{task_id}: pass@1 requires one candidate"
                )
            passed += candidates[0].get("base_status") == "pass"
        official = float(result["pass_at_k"]["base"]["pass@1"])
        empirical = passed / expected
        if abs(official - empirical) > 1e-12:
            raise ValueError(
                f"{benchmark}/{method}: official pass@1 {official} disagrees "
                f"with task statuses {empirical}"
            )
        method_summary = {
            "samples": expected,
            "base_passed": passed,
            "base_pass_at_1": official,
            "result_file": result_path.name,
        }
        task_statuses.setdefault(benchmark, {})[method] = {
            "base": {
                str(task_id): candidates[0].get("base_status") == "pass"
                for task_id, candidates in task_results.items()
            }
        }
        plus_metrics = result["pass_at_k"].get("plus")
        if plus_metrics is not None:
            plus_passed = sum(
                candidates[0].get("base_status") == "pass"
                and candidates[0].get("plus_status") == "pass"
                for candidates in task_results.values()
            )
            plus_official = float(plus_metrics["pass@1"])
            plus_empirical = plus_passed / expected
            if abs(plus_official - plus_empirical) > 1e-12:
                raise ValueError(
                    f"{benchmark}/{method}: official plus pass@1 "
                    f"{plus_official} disagrees with task statuses {plus_empirical}"
                )
            method_summary.update(
                {
                    "plus_passed": plus_passed,
                    "plus_pass_at_1": plus_official,
                }
            )
            task_statuses[benchmark][method]["plus"] = {
                str(task_id): (
                    candidates[0].get("base_status") == "pass"
                    and candidates[0].get("plus_status") == "pass"
                )
                for task_id, candidates in task_results.items()
            }
        summary["benchmarks"].setdefault(benchmark, {})[method] = method_summary
    methods = {method for by_method in task_statuses.values() for method in by_method}
    reference = next(
        (
            method
            for method in (
                "optimized_source_reuse",
                "source_reuse_eagle3",
                "naive_source_reuse",
                "native_ar",
            )
            if method in methods
        ),
        None,
    )
    if reference is not None:
        summary["reference_method"] = reference
        summary["paired_quality"] = {}
        for benchmark, by_method in sorted(task_statuses.items()):
            if reference not in by_method:
                continue
            for method, status_by_suite in sorted(by_method.items()):
                if method == reference:
                    continue
                for suite in sorted(set(by_method[reference]) & set(status_by_suite)):
                    paired = paired_binary_delta_summary(
                        by_method[reference][suite],
                        status_by_suite[suite],
                    )
                    summary["paired_quality"].setdefault(benchmark, {}).setdefault(
                        method, {}
                    )[suite] = paired
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    summary = summarize(args.output_dir)
    destination = args.output_dir / "evalplus-summary.json"
    destination.write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
