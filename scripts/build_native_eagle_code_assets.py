"""Re-score all native EAGLE code arms and regenerate scoped paper tables."""

import json
import runpy
from pathlib import Path

runpy.run_path("scripts/analyze_native_eagle_code_layers.py")
runpy.run_path("scripts/score_native_eagle_code_breadth.py", run_name="__main__")
runpy.run_path("scripts/analyze_native_eagle_code_crops.py", run_name="__main__")
root = Path("reports/autoresearch-20260907")
out = Path("paper/iclr2027/generated")
d = json.loads((root / "native-eagle-code-breadth-summary.json").read_text())
assert all(r["passed"] == 25 and r["capped"] == 0 for r in d["results"].values())
lines = [
    r"\begin{tabular}{lrrrr}",
    r"\toprule",
    r"Interface & Weights & Tokens/s & Native retained (95\% CI) & Tests passed \\",
    r"\midrule",
]
for method, label, params in [
    ("native_target_eagle3", "Released native", 83.89),
    ("relay_base", "Repacked full", 83.89),
    ("relay_two", "Layers 25,33", 33.55),
    ("relay_three17", "Layers 17,25,33", 50.33),
    ("relay_three9", "Layers 9,25,33", 50.33),
    ("relay_three1", "Layers 1,25,33", 50.33),
]:
    m = d["throughput"]["methods"][method]
    lo, hi = m["throughput_ci95"]
    lines.append(
        f"{label} & {params:.2f}M & {m['tokens_per_second']:.2f} & {100 * m['throughput_ratio']:.2f} [{100 * lo:.2f},{100 * hi:.2f}]\\% & 25/32 "
        + r"\\"
    )
lines += [r"\bottomrule", r"\end{tabular}"]
(out / "native_eagle_code_table.tex").write_text("\n".join(lines) + "\n")
crops = json.loads((root / "native-eagle-code-crops-summary.json").read_text())
lines = [
    r"\begin{tabular}{lrrr}",
    r"\toprule",
    r"Layers & Crop / native & Fitted / native & Fitted / crop (95\% CI) \\",
    r"\midrule",
]
for c in crops["results"]:
    n = c["native_reference"]["methods"]
    m = c["crop_reference"]["methods"]["relay_fitted"]
    lo, hi = m["throughput_ci95"]
    assert all(v["accuracy"] == 0.75 for v in n.values()) and all(
        v == 0 for v in c["cap_counts"].values()
    )
    lines.append(
        ",".join(map(str, c["layers"]))
        + f" & {100 * n['relay_cropped']['throughput_ratio']:.2f}\\% & {100 * n['relay_fitted']['throughput_ratio']:.2f}\\% & {m['throughput_ratio']:.3f} [{lo:.3f},{hi:.3f}] "
        + r"\\"
    )
lines += [r"\bottomrule", r"\end{tabular}"]
(out / "native_eagle_code_crops_table.tex").write_text("\n".join(lines) + "\n")
