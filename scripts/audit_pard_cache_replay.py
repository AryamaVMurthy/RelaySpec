"""Audit controlled PARD replay artifacts and expose the numerical interventions."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate_path = args.run / "cache-replay-gate.json"
    before = gate_path.read_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        for path in args.run.glob("cache-replay-rank*.json"):
            shutil.copy2(path, Path(tmp) / path.name)
        subprocess.run(
            [sys.executable, "scripts/summarize_pard_cache_replay.py"],
            env={**os.environ, "RELAYSPEC_OUTPUT": tmp},
            check=True,
            capture_output=True,
        )
        if (Path(tmp) / gate_path.name).read_bytes() != before:
            raise ValueError("replay gate does not reproduce from its workers")
    declaration_path = Path("configs/submission/baselines/pard-cache-replay.json")
    declaration = json.loads(declaration_path.read_text())
    inputs = {
        str(gate_path): digest(gate_path),
        str(declaration_path): digest(declaration_path),
    }
    table = []
    cache_comparison = None
    for rank in range(4):
        path = args.run / f"cache-replay-rank{rank}.json"
        row = json.loads(path.read_text())
        inputs[str(path)] = digest(path)
        if row["source_rows_sha256"] != declaration["source_rows_sha256"]:
            raise ValueError("diagnostic replay used another failing request")
        cache_comparison = cache_comparison or row["committed_cache_comparison"]
        if cache_comparison != row["committed_cache_comparison"]:
            raise ValueError("workers did not reproduce the same cache comparison")
        for cache, values in row["comparisons"].items():
            for mode in ("serial", "block"):
                score = values[mode]
                ids, logits = score["top_ids"][-1], score["top_scores"][-1]
                choice = score["argmax_ids"][-1]
                lookup = dict(zip(ids, logits, strict=True))
                if choice not in lookup or lookup[choice] != max(logits):
                    raise ValueError("reported target choice is not a leading logit")
                table.append(
                    {
                        "query_length": row["query_length"],
                        "cache": cache,
                        "evaluation": mode,
                        "argmax_id": choice,
                        "top_ids": ids,
                        "top_scores": logits,
                        "causal_prefix_exact": values["causal_prefix_exact"],
                    }
                )
    output = {
        "status": "characterized",
        "input_sha256": inputs,
        "source_rows_sha256": declaration["source_rows_sha256"],
        "problem_id": declaration["problem_id"],
        "generated_token_index": declaration["first_difference"],
        "committed_cache_comparison": cache_comparison,
        "interventions": table,
        "original_exact_ar_pilot_status": "failed",
        "scope": "All preceding AR argmaxes and PARD top-five traces replay exactly. "
        "Original AR and original-shape PARD final scores reproduce. Cache copies "
        "are checked tensorwise before every intervention. Controlled query-shape "
        "and cache effects are demonstrated at one existing development prefix. "
        "No exact CUDA-kernel attribution, task-quality claim, or timing comparison. "
        "The original exact-AR gate remains failed.",
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
