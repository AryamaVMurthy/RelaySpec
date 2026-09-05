"""Require both duplicate public-objective fits and every SD-square pilot invariant."""

import hashlib
import json
import os
from pathlib import Path

from relayspec.sd_square_adapter import committed_tokens


def main():
    output = Path(os.environ["RELAYSPEC_OUTPUT"])
    fits, decoding, inputs = [], [], {}
    for rank in range(4):
        for prefix in ("pilot", "decoding", "training"):
            path = output / f"{prefix}-rank{rank}.json"
            inputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        fit = json.loads((output / f"pilot-rank{rank}.json").read_text())
        row = json.loads((output / f"decoding-rank{rank}.json").read_text())
        if (
            fit["status"] != "pass"
            or fit["rank"] != rank
            or fit["distinct_records_seen"] != 4
            or fit["training"]
            != json.loads((output / f"training-rank{rank}.json").read_text())
            or not all(
                fit[k]
                for k in (
                    "zero_guidance_bit_identical",
                    "steering_reload_bit_identical",
                    "optimizer_reload_bit_identical",
                    "frozen_parameter_versions_and_storage_unchanged",
                    "ar_exact",
                )
            )
            or row["output_ids"] != row["ar_ids"]
        ):
            raise ValueError("SD-square worker invariant failed")
        counted, raw = committed_tokens(row["trace"], 64, 151645)
        if (
            counted != row["output_ids"]
            or raw != row["raw_committed_ids"]
            or any(
                block["tokens"] != block["verifier_argmax"] for block in row["trace"]
            )
        ):
            raise ValueError("SD-square accepted-token provenance differs")
        fits.append(fit)
        decoding.append(row)
    for a, b in ((0, 1), (2, 3)):
        if any(
            fits[a][key] != fits[b][key]
            for key in (
                "trainable_sha256",
                "frozen_drafter_sha256",
                "objective",
                "ordered_pool_files",
                "seed",
            )
        ):
            raise ValueError("duplicate SD-square fitting changed weights or data")
        if [(r["loss"], r["gradient_norm"]) for r in fits[a]["training"]] != [
            (r["loss"], r["gradient_norm"]) for r in fits[b]["training"]
        ]:
            raise ValueError("duplicate SD-square training losses/gradients differ")
        if any(
            decoding[a][key] != decoding[b][key]
            for key in ("input_ids", "output_ids", "trace")
        ):
            raise ValueError("duplicate SD-square decoding differs")
    (output / "sd-square-pilot-gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "input_sha256": inputs,
                "fitting": fits,
                "scope": fits[0]["scope"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
