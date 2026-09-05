#!/usr/bin/env python3
"""Display the dependency-ready research work without launching it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relayspec.submission_plan import PlanError, validate_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        plan = json.loads((root / "configs/submission/plan.json").read_text())
        status = validate_plan(plan, root)
    except (PlanError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Plan invalid: {exc}")
        return 1
    print("Plan structure, issue coverage and referenced files are consistent.")
    print("This check does not establish scientific validity or submission acceptance.")
    titles = {task["id"]: task["title"] for task in plan["tasks"]}
    print("Next dependency-ready tasks:")
    for task_id in status["next_tasks"]:
        print(f"  {task_id}: {titles[task_id]}")
    remaining = status["remaining_required"]
    print(f"Required tasks without completion evidence: {len(remaining)}")
    if remaining:
        print("  " + ", ".join(remaining))
    if args.require_complete and remaining:
        print("Submission execution is incomplete.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
