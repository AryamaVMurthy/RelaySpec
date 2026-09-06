"""Freeze completed Numina small-to-large evidence for the manuscript."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/data-small-20260906"
PAPER = ROOT / "paper/iclr2027"


def main():
    result = json.loads((BASE / "results.json").read_text())
    if result["status"] != "complete":
        raise ValueError("Completed evidence required")
    for name, expected in result["source_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError("Evidence source changed")
    gate = json.loads((BASE / "run-28339/completion-gate.json").read_text())
    if gate["status"] != "pass" or gate["records"] != 1792:
        raise ValueError("Full paired evaluation required")
    values = result["panels"]["fixed_8192_updates"]
    best = max(v["tokens_per_second"] for v in values.values())
    lines = [
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"Records & Tokens/s & Speedup / AR & Correct / 128 \\",
        r"\midrule",
    ]
    for n, v in values.items():
        lines.append(
            f"{int(n):,} & {v['tokens_per_second']:.2f} & {v['throughput_ratio']:.2f}$\\times$ & {round(128 * v['accuracy'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "generated/expanded_numina_data_table.tex").write_text(
        "\n".join(lines) + "\n"
    )
    shutil.copy2(
        BASE / "distinct-record-scaling.pdf", PAPER / "figures/expanded_numina_data.pdf"
    )
    registry = {
        "source": str(BASE),
        "source_sha256": result["source_sha256"],
        "retention_512": values["512"]["tokens_per_second"] / best,
        "scope": result["scope"],
    }
    (PAPER / "generated/expanded_numina_data_provenance.json").write_text(
        json.dumps(registry, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
