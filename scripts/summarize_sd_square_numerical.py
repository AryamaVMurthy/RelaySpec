"""Require exact duplicate numerical replays, without relabeling failed AR identity."""

import hashlib
import json
import os
from pathlib import Path


def main():
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    rows, inputs = [], {}
    for rank in range(4):
        path = output / f"numerical-rank{rank}.json"
        row = json.loads(path.read_text())
        if row["status"] != "pass" or row["rank"] != rank:
            raise ValueError("SD-square numerical worker failed")
        inputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(row)
    for a, b in ((0, 1), (2, 3)):
        if {k: v for k, v in rows[a].items() if k != "rank"} != {
            k: v for k, v in rows[b].items() if k != "rank"
        }:
            raise ValueError("SD-square numerical duplicate differed")
    (output / "sd-square-numerical-gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "input_sha256": inputs,
                "first_differences": {
                    rows[r]["target_attention"]: rows[r]["first_difference"]
                    for r in (0, 2)
                },
                "public_ar_matches_correct_cached_ar": {
                    rows[r]["target_attention"]: rows[r][
                        "public_ar_matches_correct_cached_ar"
                    ]
                    for r in (0, 2)
                },
                "original_ar_identity_gate": "failed_preserved",
                "scope": "Numerical diagnosis only. Same effective token prefixes and duplicate observations "
                "do not restore exact-AR equivalence or establish full-answer quality. No new fitting.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
