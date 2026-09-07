"""Regenerate matched native EAGLE budget and workload controls."""

import runpy
from pathlib import Path

for name in [
    "native_eagle_initialization",
    "native_eagle_budget",
    "native_eagle_budget_transfer",
]:
    namespace = runpy.run_path(f"scripts/analyze_{name}.py")
    if name == "native_eagle_budget_transfer":
        data = namespace["result"]
lines = [
    r"\begin{tabular}{lrrrrrr}",
    r"\toprule",
    r"Task & Crop & Rnd128 & Inh128 & Rnd8192 & Inh8192 & SVD \\",
    r"\midrule",
]
methods = [
    "relay_cropped",
    "relay_random_u128",
    "relay_inherited_u128",
    "relay_random_u8192",
    "relay_inherited_u8192",
    "relay_svd1536",
]
for cell in data["results"]:
    values = cell["full_native_reference"]["methods"]
    lines.append(
        cell["task"]
        + " & "
        + " & ".join(f"{100 * values[m]['throughput_ratio']:.1f}" for m in methods)
        + r" \\"
    )
lines += [r"\bottomrule", r"\end{tabular}"]
Path("paper/iclr2027/generated/native_eagle_budget_table.tex").write_text(
    "\n".join(lines) + "\n"
)
