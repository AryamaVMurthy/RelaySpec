from __future__ import annotations

import argparse
import json
from pathlib import Path

from evalplus.data import get_human_eval_plus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    problems = get_human_eval_plus()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for task_id in sorted(problems):
            problem = problems[task_id]
            sample = {
                "task_id": task_id,
                "solution": problem["prompt"] + problem["canonical_solution"],
            }
            stream.write(json.dumps(sample) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
