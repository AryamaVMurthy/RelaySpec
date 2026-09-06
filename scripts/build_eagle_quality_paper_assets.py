"""Export the complete EAGLE quality comparison after replaying both raw shards."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from relayspec.cached_fit_evidence import digest


def quality_table(result):
    if (
        result.get("status") != "complete"
        or result.get("stage") != "full"
        or result.get("requests") != 128
        or result.get("rows") != 1408
    ):
        raise ValueError("paper export requires all 128 requests and eleven methods")
    labels = [
        ("native_ar", "AR", "--"),
        ("native_target_eagle3", "Native drafter", "--"),
        ("source_reuse_eagle3", "Source reuse", "--"),
    ]
    labels += [
        (f"relay_eagle3_{kind}_n{n}", label, str(n))
        for kind, label in [
            ("dense", "Dense"),
            ("factorized2048", "Linear 2048"),
            ("mlp2048", "MLP 2048"),
        ]
        for n in [512, 2048]
    ]
    labels += [
        ("relay_eagle3_factorized512_n2048", "Linear 512", "2048"),
        ("relay_eagle3_mlp512_n2048", "MLP 512", "2048"),
    ]
    reference = "relay_eagle3_dense_n2048"
    paired = result["comparisons"][reference]["paired_throughput"]["methods"]
    if {name for name, _, _ in labels} != set(result["methods"]) or set(paired) != set(
        result["methods"]
    ):
        raise ValueError("paper table differs from the complete method set")
    table = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & $N$ & Tokens/s & Dense retained (\%) [95\% CI] & Correct & Cap \\",
        r"\midrule",
    ]
    for name, label, n in labels:
        row, quality = paired[name], result["methods"][name]
        lo, hi = [100 * x for x in row["throughput_ci95"]]
        table.append(
            f"{label} & {n} & {row['tokens_per_second']:.2f} & "
            f"{100 * row['throughput_ratio']:.2f} [{lo:.2f}, {hi:.2f}] & "
            f"{quality['correct_count']}/128 & {quality['cap_length_outputs']}" + r" \\"
        )
    return "\n".join(table + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("paper/iclr2027"))
    args = parser.parse_args()
    base = Path("reports/mapper-scaling-20260905")
    registry = base / "eagle3-small-quality-results.json"
    result = json.loads(registry.read_text())
    ledger_path = base / "eagle3-small-quality-full/jobs.json"
    ledger = json.loads(ledger_path.read_text())
    jobs = sorted(ledger["jobs"], key=lambda job: job["stage"])
    if [job["stage"] for job in jobs] != ["full0", "full1"]:
        raise ValueError("quality requires the two exact full shards")
    with tempfile.TemporaryDirectory() as directory:
        expected = Path(directory) / "result.json"
        subprocess.run(
            [
                sys.executable,
                "scripts/audit_eagle_quality.py",
                "--protocol",
                "configs/submission/scaling/eagle3-small-quality-v1/protocol.json",
                "--stage",
                "full",
                "--runs",
                *[str(args.raw_root / job["local"]) for job in jobs],
                "--ledger",
                str(ledger_path),
                "--scoring-repo",
                str(args.raw_root),
                "--pilot-result",
                str(base / "eagle3-small-quality-pilot-results.json"),
                "--output",
                str(expected),
            ],
            check=True,
            capture_output=True,
        )
        if json.loads(expected.read_text()) != result:
            raise ValueError("EAGLE quality evidence does not reproduce")
    table = quality_table(result)
    generated = args.output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "eagle_quality_table.tex").write_text(table)
    (generated / "eagle_quality_evidence_registry.json").write_text(
        json.dumps(
            {
                "raw_root": str(args.raw_root.resolve()),
                "input_sha256": {
                    str(registry): digest(registry),
                    **result["input_sha256"],
                },
                "scope": result["scope"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
