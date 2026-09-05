#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from relayspec.metrics import paired_mismatch_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-method", required=True)
    parser.add_argument("--candidate-method", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--context-chars", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for path in args.input
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    audit = paired_mismatch_audit(
        rows,
        reference_method=args.reference_method,
        candidate_method=args.candidate_method,
        context_chars=args.context_chars,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Paired finite-precision mismatch audit",
        "",
        f"Reference: `{args.reference_method}`. Candidate: `{args.candidate_method}`.",
        "",
        (
            f"Agreement: **{audit['paired_requests'] - audit['mismatch_count']}/"
            f"{audit['paired_requests']} ({audit['match_rate']:.2%})**."
        ),
        "",
        "Every mismatch is retained below with independently scored task outcomes.",
        "",
    ]
    for mismatch in audit["mismatches"]:
        lines.extend(
            [
                f"## {mismatch['problem_id']}",
                "",
                (
                    f"Common character prefix: {mismatch['common_prefix_chars']}; "
                    f"reference/candidate correct: {mismatch['reference_correct']}/"
                    f"{mismatch['candidate_correct']}."
                ),
                "",
                f"- Reference context: `{mismatch['reference_tail']!r}`",
                f"- Candidate context: `{mismatch['candidate_tail']!r}`",
                "",
            ]
        )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
