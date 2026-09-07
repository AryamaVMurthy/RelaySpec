"""Render a bounded interpretation of scalar and low-rank calibration controls."""

import json
import runpy
from pathlib import Path

root = Path("reports/autoresearch-20260907")
for name in [
    "analyze_native_residual",
    "analyze_native_residual_seeds",
    "analyze_native_residual_transfer",
    "analyze_native_residual_seed_transfer",
]:
    runpy.run_path(f"scripts/{name}.py")
fits = []
for filename, lanes in [
    ("native-residual-summary.json", [3]),
    ("native-residual-seeds-summary.json", [1, 3]),
]:
    data = json.loads((root / filename).read_text())
    fits.extend(data["results"][i] for i in lanes)
assert {r["fit"]["trial"]["seed"] for r in fits} == {1729, 1730, 1731}
assert all(
    r["fit"]["trial"]["native_residual_rank"] == 128
    and r["fit"]["trial"]["learning_rate"] == 0.006
    for r in fits
)
ratios = []
for filename, lanes in [
    ("native-residual-transfer-summary.json", [3]),
    ("native-residual-seed-transfer-summary.json", [1, 3]),
]:
    data = json.loads((root / filename).read_text())
    for cell in data["results"]:
        for lane in lanes:
            summary = cell["full_fit_reference"]
            m = summary["methods"][f"relay_residual_lane{lane}"]
            assert m["token_matches"] == summary["requests"]
            ratios.append(m["throughput_ratio"])
assert len(ratios) == 12
losses = [r["fit"]["validation_objective"] for r in fits]
full_fit = json.loads((root / "native-initialization-budget-summary.json").read_text())[
    "results"
][2]["fit"]
assert (
    full_fit["trial"]["distinct_examples"] == 128 and full_fit["trial"]["steps"] == 128
)
input_width, output_width = (
    fits[0]["fit"]["input_width"],
    fits[0]["fit"]["output_width"],
)
trained = 128 * (input_width + output_width)
deployed = input_width * output_width
assert trained == 1572864 and deployed == 33554432
paragraph = r"""\paragraph{A low-rank correction between gains and full fitting.}
Freezing the inherited columns and fitting an additive low-rank correction
provides another calibration control, using the standard low-rank
adaptation construction \citep{hu2022lora}. Rank 128 trains 1.57M parameters
instead of 33.55M, a 95.3\% reduction in trainable weights. Folding the
correction retains the same dense deployed projection, so this is not
an additional inference-memory saving. At 128 records and 128 updates,
the higher of two tested rates (0.006) retains RANGE\% of full-matrix
fitting throughput across three fitting seeds and four workloads.
Validation error remains LOSS, versus FULLLOSS for full fitting.
The individual paired intervals and every rank/rate arm are retained in
the artifacts. These are the same eight exposed requests per workload,
with capped MATH and dialogue outputs. The range describes twelve
seed/workload point estimates, not a confidence bound or a new independent
benchmark sample. The result supports a parameter-efficient approximation
to full calibration, with seed and workload dependence, rather than exact
equivalence or a novel low-rank algorithm.
"""
paragraph = (
    paragraph.replace("RANGE", f"{100 * min(ratios):.1f}--{100 * max(ratios):.1f}")
    .replace("FULLLOSS", f"{full_fit['validation_objective']:.3f}")
    .replace("LOSS", f"{min(losses):.3f}--{max(losses):.3f}")
)
Path("paper/iclr2027/generated/native_correction_paragraph.tex").write_text(paragraph)
