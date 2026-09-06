"""Build native initialization evidence with untrained and matched-budget controls."""

import runpy
from pathlib import Path

root = Path("reports/autoresearch-20260907")
for script in [
    "analyze_native_initialization",
    "analyze_native_initialization_budget",
    "analyze_native_short_controls",
    "analyze_native_subsets",
]:
    runpy.run_path(f"scripts/{script}.py")
analyze = runpy.run_path("scripts/analyze_native_initialization_transfer.py")["analyze"]
studies = []
for job, wave, steps, name in [
    (28533, 43, 8192, "native-initialization-transfer-summary"),
    (28537, 46, 128, "native-short-transfer-summary"),
]:
    data = analyze(job, wave, root / f"{name}.json")
    studies.append((steps, data))
lines = [
    r"\begin{tabular}{lrrrrrrr}",
    r"\toprule",
    r"Task & Updates & Crop & Rnd16 & Inh16 & Rnd128 & Inh128 & Ref512 \\",
    r"\midrule",
]
methods = [
    "relay_cropped",
    "relay_n16_random",
    "relay_n16_native_columns",
    "relay_n128_random",
    "relay_n128_native_columns",
    "relay_two",
]
for steps, data in studies:
    for cell in data["results"]:
        summary = cell["full_native_reference"]
        assert all(
            m["token_matches"] == summary["requests"]
            for m in summary["methods"].values()
        )
        values = " & ".join(
            f"{100 * summary['methods'][m]['throughput_ratio']:.1f}" for m in methods
        )
        lines.append(f"{cell['task']} & {steps:,} & {values} " + r"\\")
    if steps == 8192:
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
Path("paper/iclr2027/generated/native_initialization_table.tex").write_text(
    "\n".join(lines) + "\n"
)
paragraph = r"""\paragraph{Inherited weights change the calibration requirement.}
A native interface permits an additional control: retain the released
projection columns for layers 25 and 33, discarding the other columns.
This untrained crop already preserves much of native throughput
(Table~\ref{tab:native-initialization}). Initializing the same compact map
from these columns instead of random weights improves sparse-data fitting.
At 128 updates on 16 records, validation error falls from 0.347 to 0.038
and GSM8K throughput retention rises from 51.7\% to 99.3\%, relative to
the original 512-record compact map. At 128 records the corresponding
retentions are 72.4\% and 101.0\%. All non-initialization fitting settings
are matched. The two 128-update inherited fits take 1.55--1.58 seconds
of optimizer-update time and 8.8--10.4 seconds within the fitting worker,
excluding frozen-feature extraction and original drafter training. Longer inherited fits at 1,024 and 8,192 updates give no
clear GSM8K throughput advantage in this screen.
Table~\ref{tab:native-initialization} pairs the short and long fit candidates
with untrained cropping across four workloads. This separates inherited
performance from the additional effect of calibration. All paired outputs
match, but MATH and dialogue include capped outputs, and code and dialogue
quality are unscored. Each workload has eight exposed requests, with two
turns per dialogue conversation. The fixed record order and copied
initialization produce identical weight storage when the nominal fitting
seed changes, so those repeats are not independent fitting-seed evidence.
Four additional disjoint 16-record training windows, verified against
the frozen cache index, retain 99.1--100.0\% of the 512-record compact
reference after 128 updates on the same GSM8K screen. This checks
calibration-record sensitivity without adding independent test questions.
This is an initialization-dependent calibration finding, not a universal
minimum-data claim or a new general pruning algorithm.
"""
Path("paper/iclr2027/generated/native_initialization_paragraph.tex").write_text(
    paragraph
)
