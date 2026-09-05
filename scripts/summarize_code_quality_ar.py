"""Paired AR-relative quality from saved official EvalPlus task outcomes."""

import argparse
import hashlib
import json
from pathlib import Path

from relayspec.metrics import paired_binary_delta_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparisons = []
    hashes = {}
    for size in [8, 14]:
        root = args.reports_root / f"eagle3-{size}b-code-quality"
        summary = json.loads((root / "evalplus-summary.json").read_text())
        for task, methods in summary["benchmarks"].items():
            results = {}
            for method in ["native_ar", "relay_eagle3", "source_reuse_eagle3"]:
                path = root / "evalplus" / methods[method]["result_file"]
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                result = json.loads(path.read_text())["eval"]
                if len(result) != methods[method]["samples"] or any(
                    len(v) != 1 for v in result.values()
                ):
                    raise ValueError(
                        "quality outcome count does not match the declared pass@1 sample set"
                    )
                results[method] = result
            for suite in ["base", "plus"]:
                values = {
                    method: {
                        key: candidates[0]["base_status"] == "pass"
                        and (suite == "base" or candidates[0]["plus_status"] == "pass")
                        for key, candidates in result.items()
                    }
                    for method, result in results.items()
                }
                comparison = paired_binary_delta_summary(
                    values["native_ar"], values["relay_eagle3"]
                )
                comparisons.append(
                    {
                        "target_billions": size,
                        "benchmark": task,
                        "suite": suite,
                        **comparison,
                        "relay_source_task_outcomes_identical": values["relay_eagle3"]
                        == values["source_reuse_eagle3"],
                    }
                )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "code-quality-ar.json").write_text(
        json.dumps(
            {
                "comparisons": comparisons,
                "source_sha256": hashes,
                "scope": "One completion per task. Paired task bootstrap, no fitting-seed uncertainty or multiple-comparison correction. Identical observed outcomes yield a degenerate empirical bootstrap; this is not proof of equality or noninferiority.",
            },
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# EAGLE-3 saved-output code quality",
        "",
        "Official EvalPlus base and plus test cases; paired against the same run’s AR outputs.",
        "",
        "| Target | Benchmark | Suite | AR passes | Relay passes | Delta (pp) | Paired 95% interval (pp) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in comparisons:
        n = row["paired_tasks"]
        lo, hi = row["accuracy_delta_ci95"]
        lines.append(
            f"| {row['target_billions']}B | {row['benchmark']} | {row['suite']} | {round(n * row['reference_accuracy'])}/{n} | {round(n * row['candidate_accuracy'])}/{n} | {100 * row['accuracy_delta']:+.2f} | [{100 * lo:+.2f}, {100 * hi:+.2f}] |"
        )
    lines += [
        "",
        "Relay and source reuse have identical per-task pass/fail outcomes in all eight cells. Keep the 8B regressions visible. These are retrospective, single-fit results; paired task intervals do not capture fitting-seed variation. A zero-width empirical bootstrap from identical outcomes does not establish population equality or noninferiority.",
        "",
    ]
    (args.output / "CODE_QUALITY.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
