from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model", action="append", nargs=2, metavar=("ID", "REVISION"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model:
        raise ValueError("at least one pinned model is required")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for repo_id, revision in args.model:
        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=args.cache_dir,
        )
        print(
            json.dumps(
                {"model": repo_id, "revision": revision, "snapshot_path": path},
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
