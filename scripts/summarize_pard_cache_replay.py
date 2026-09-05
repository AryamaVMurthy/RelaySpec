"""Collect all four controlled cache/query-shape diagnostic workers."""

import hashlib
import json
import os
from pathlib import Path


def main():
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    declaration = Path("configs/submission/baselines/pard-cache-replay.json")
    expected = json.loads(declaration.read_text())
    expected_hash = hashlib.sha256(declaration.read_bytes()).hexdigest()
    workers = []
    for rank, width in enumerate(expected["query_lengths"]):
        path = output / f"cache-replay-rank{rank}.json"
        row = json.loads(path.read_text())
        if (
            row["status"] != "pass"
            or row["rank"] != rank
            or row["query_length"] != width
            or row["input_sha256"][str(declaration)] != expected_hash
            or not row["ar_replay_argmax_exact"]
            or not row["pard_replay_top5_exact"]
            or not row["ar_final_top5_exact"]
            or (width == 13 and row["pard_original_shape_final_top5_exact"] is not True)
            or not all(
                v["cache_copies_verified_exact"] and v["causal_prefix_exact"]
                for v in row["comparisons"].values()
            )
        ):
            raise ValueError("controlled replay did not pass every declared check")
        workers.append(
            {
                "rank": rank,
                "query_length": width,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    result = {
        "status": "pass",
        "workers": workers,
        "declaration_sha256": expected_hash,
        "scope": expected["scope"],
    }
    (output / "cache-replay-gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
