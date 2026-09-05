from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from relayspec.evaluation import score_math_completion
from relayspec.metrics import paired_bootstrap_summary, paired_mismatch_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--qwen-math-eval",
        type=Path,
        help="Path to a pinned Qwen2.5-Math evaluation directory.",
    )
    parser.add_argument(
        "--subset-manifest",
        type=Path,
        help="Optional manifest whose problem IDs define a preregistered subset.",
    )
    parser.add_argument("--output-prefix", default="benchmark")
    return parser.parse_args()


def load_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("benchmark-rank*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    if not rows:
        raise ValueError(f"no benchmark rows found in {output_dir}")
    return rows


def build_math_scorer(qwen_math_eval: Path | None):
    if qwen_math_eval is None:

        def internal(completion: str, reference: Any, benchmark: str):
            if benchmark not in {"math500", "gsm8k"}:
                return {"predicted_answer": None, "correct": None}
            return score_math_completion(completion, reference)

        return internal, "relayspec_internal"
    evaluation_dir = qwen_math_eval.resolve()
    latex_dir = evaluation_dir / "latex2sympy"
    sys.path[:0] = [str(evaluation_dir), str(latex_dir)]
    parser = importlib.import_module("parser")
    grader = importlib.import_module("grader")

    def official(completion: str, reference: Any, benchmark: str) -> dict[str, Any]:
        if benchmark not in {"math500", "gsm8k"}:
            return {"predicted_answer": None, "correct": None}
        data_name = "math" if benchmark == "math500" else "gsm8k"
        prediction = parser.extract_answer(completion, data_name)
        return {
            "predicted_answer": prediction,
            "correct": bool(grader.math_equal(prediction, str(reference))),
        }

    return official, "qwen2.5_math_official"


def main() -> None:
    args = parse_args()
    rows = load_rows(args.output_dir)
    if args.subset_manifest is not None:
        subset = json.loads(args.subset_manifest.read_text(encoding="utf-8"))
        expected_ids = {str(row["problem_id"]) for row in subset["records"]}
        rows = [row for row in rows if str(row["problem_id"]) in expected_ids]
        methods_found = {str(row["method"]) for row in rows}
        for method in methods_found:
            observed_ids = {
                str(row["problem_id"]) for row in rows if str(row["method"]) == method
            }
            if observed_ids != expected_ids:
                raise ValueError(
                    f"subset rows for {method} do not match subset manifest"
                )
        if not methods_found:
            raise ValueError("subset manifest selected no benchmark rows")
    config = yaml.safe_load(
        (args.output_dir / "config.yaml").read_text(encoding="utf-8")
    )
    token_cap = int(config["generation"]["max_new_tokens"])
    scorer, scorer_name = build_math_scorer(args.qwen_math_eval)
    scored: list[dict[str, Any]] = []
    for row in rows:
        scored.append(
            {
                **row,
                "hit_token_cap": int(row["output_tokens"]) >= token_cap,
                **scorer(
                    row["completion"],
                    row["reference_answer"],
                    row["benchmark"],
                ),
            }
        )
    methods = tuple(dict.fromkeys(str(row["method"]) for row in scored))
    reference_method = "native_ar" if "native_ar" in methods else methods[0]
    summary = paired_bootstrap_summary(
        scored,
        methods=methods,
        reference_method=reference_method,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    summary["scorer"] = scorer_name
    mismatch_audits = {
        method: paired_mismatch_audit(
            scored,
            reference_method=reference_method,
            candidate_method=method,
        )
        for method in methods
        if method != reference_method
    }
    summary["finite_precision_agreement"] = {
        method: {
            "mismatch_count": audit["mismatch_count"],
            "match_rate": audit["match_rate"],
        }
        for method, audit in mismatch_audits.items()
    }
    summary["by_benchmark"] = {}
    for benchmark in sorted({str(row["benchmark"]) for row in scored}):
        benchmark_rows = [row for row in scored if row["benchmark"] == benchmark]
        summary["by_benchmark"][benchmark] = paired_bootstrap_summary(
            benchmark_rows,
            methods=methods,
            reference_method=reference_method,
            samples=args.bootstrap_samples,
            seed=args.seed,
        )
    scored_path = args.output_dir / f"{args.output_prefix}-scored.jsonl"
    with scored_path.open("w", encoding="utf-8") as stream:
        for row in scored:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    summary_path = args.output_dir / f"{args.output_prefix}-paper-summary.json"
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    mismatch_path = args.output_dir / f"{args.output_prefix}-mismatch-audit.json"
    mismatch_path.write_text(
        json.dumps(mismatch_audits, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
