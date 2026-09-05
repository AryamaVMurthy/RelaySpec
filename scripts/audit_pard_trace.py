"""Verify saved PARD acceptance steps and characterize the first AR differences."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs, differences, requests = {}, [], []
    cycles = 0
    for rank in range(4):
        path = args.run / f"pard-rank{rank}.json"
        gate_path = args.run / f"pard-gate-rank{rank}.json"
        gate = json.loads(gate_path.read_text())
        if (
            gate["rank"] != rank
            or gate["rows_sha256"] != digest(path)
            or not gate["diagnostic_mode"]
        ):
            raise ValueError("missing source-hashed diagnostic worker")
        inputs.update({str(path): digest(path), str(gate_path): digest(gate_path)})
        rows = json.loads(path.read_text())["rows"]
        ar, pard, duplicate = [rows[k] for k in ("eager_ar", "pard", "pard_duplicate")]
        requests.append(pard["problem_id"])
        for key in (
            "token_ids",
            "raw_token_ids",
            "accepted_lengths",
            "target_calls",
            "draft_calls",
            "input_ids",
        ):
            if pard[key] != duplicate[key]:
                raise ValueError("duplicate PARD generation differs")
        if pard["input_ids"] != ar["input_ids"]:
            raise ValueError("PARD and AR received different prompts")
        for row in (pard, duplicate):
            emitted = []
            if len(row["target_trace"]) != len(row["accepted_lengths"]):
                raise ValueError("missing target-call trace")
            for trace, n in zip(
                row["target_trace"], row["accepted_lengths"], strict=True
            ):
                width = len(trace["argmax_ids"]) - 1
                prefix = row["input_ids"] + emitted
                if (
                    trace["output_start"] != len(emitted)
                    or trace["prefix_ids"] != prefix
                    or not 1 <= n <= width + 1
                    or trace["incoming_ids"][-width:][: n - 1]
                    != trace["argmax_ids"][: n - 1]
                    or trace["cache_position"][-width - 1 :]
                    != list(range(len(prefix) - 1, len(prefix) + width))
                ):
                    raise ValueError(
                        "invalid recorded acceptance prefix or cache position"
                    )
                emitted.extend(trace["argmax_ids"][:n])
                cycles += 1
            if emitted != row["raw_token_ids"]:
                raise ValueError("emitted tokens differ from verified target argmaxes")
        first = next(
            (
                i
                for i, (a, b) in enumerate(zip(ar["token_ids"], pard["token_ids"]))
                if a != b
            ),
            None,
        )
        if first is None:
            if ar["token_ids"] != pard["token_ids"]:
                raise ValueError("uncharacterized output-length difference")
            continue
        calls = [
            (t, n)
            for t, n in zip(pard["target_trace"], pard["accepted_lengths"], strict=True)
            if t["output_start"] <= first < t["output_start"] + n
        ]
        if len(calls) != 1:
            raise ValueError("first difference has no unique verification call")
        trace, _ = calls[0]
        offset = first - trace["output_start"]
        a = ar["target_trace"][first]
        p = {
            "argmax_id": trace["argmax_ids"][offset],
            "top_ids": trace["top_ids"][offset],
            "top_scores": trace["top_scores"][offset],
        }
        if (
            trace["prefix_ids"]
            != ar["input_ids"] + ar["token_ids"][: trace["output_start"]]
        ):
            raise ValueError("first-difference call has another committed prefix")
        choices = [ar["token_ids"][first], pard["token_ids"][first]]
        for observed in (a, p):
            scores = dict(zip(observed["top_ids"], observed["top_scores"], strict=True))
            if not set(choices).issubset(scores) or scores[
                observed["argmax_id"]
            ] != max(scores.values()):
                raise ValueError(
                    "first difference is not explained by recorded leading logits"
                )
        differences.append(
            {
                "rank": rank,
                "problem_id": pard["problem_id"],
                "output_index": first,
                "same_token_prefix": True,
                "ar": a,
                "pard": p,
                "pard_call_output_start": trace["output_start"],
                "pard_call_offset": offset,
            }
        )
    if len(set(requests)) != 4:
        raise ValueError("four distinct diagnostic requests required")
    args.output.write_text(
        json.dumps(
            {
                "status": "characterized",
                "input_sha256": inputs,
                "requests": 4,
                "exact_ar_requests": 4 - len(differences),
                "verified_acceptance_cycles": cycles,
                "all_duplicate_outputs_and_trajectories_match": True,
                "first_differences": differences,
                "scope": "Every emitted raw token follows the recorded target argmax and accepted "
                "proposal prefix. Cache positions and committed token prefixes are checked. "
                "Observed first AR differences coincide with leading-score changes under "
                "the same token prefix. This does not directly compare cache tensor values "
                "or establish task-quality equivalence. Original exact-AR pilot failure is "
                "preserved. Diagnostic timings are not valid performance measurements.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
