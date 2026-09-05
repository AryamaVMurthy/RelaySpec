# RelaySpec Main-Paper Visual Density Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild the RelaySpec manuscript as a dense but readable official-format ICLR paper with six evidence-first main figures, a nine-page main body, complete appendix tables, and fully reproducible values from recorded experiment artifacts.

**Architecture:** Keep `scripts/build_iclr_paper_assets.py` as the only source of generated paper values, tables, and vector figures. Add small loaders for paired request latency and design-selection evidence, test every derived value before plotting it, then rewrite only the manuscript presentation around the generated assets. Preserve the official ICLR style and all scientific claims while moving duplicated tables and procedural detail from the main body to the appendix.

**Tech Stack:** Python 3.12, JSON and JSONL experiment artifacts, Matplotlib, NumPy, LaTeX with the official ICLR 2027 style, pytest, Ruff, Poppler, Ghostscript, and ImageMagick.

---

## Execution constraints

- Work in `/home/aryamavmurthy/work/Latent-Recurrent` on `main`. This is an explicit exception to the usual worktree recommendation because the user requested that RelaySpec live on `main` only.
- Do not stage or modify unrelated dirty files. Stage only the exact files listed in each task.
- Do not modify `paper/iclr2027/iclr2027_conference.sty` or `paper/iclr2027/iclr2027_conference.bst`.
- Do not reduce margins, global font size, line spacing, or paper size. Do not use negative vertical spacing to evade the page limit.
- All main claims must come from `reports/final/` or the registered design-selection artifacts listed below. No new benchmark runs are required.
- Use TDD for every new derived quantity. A plot is not evidence that its underlying calculation is correct.
- Keep final plot text at approximately 7.5 pt or larger at the manuscript insertion size. Use shape, fill, hatch, or line style in addition to color.

## Evidence files used by this change

- `reports/final/{dflash,eagle3}-{8b,14b}-math500/benchmark-rank{0,1,2,3}.jsonl`
- `reports/final/{dflash,eagle3}-{8b,14b}-math500/benchmark-paper-summary.json`
- `reports/final/{dflash,eagle3}-{8b,14b}-math500/math500-confirmatory-paper-summary.json`
- `reports/final/BREADTH_MATRIX.json`
- `reports/final/EAGLE3_{8B,14B}_MEMORY.json`
- `reports/design-selection/dflash/objective-selection.json`
- `reports/design-selection/eagle3/normalized-linear-analysis.json`
- `reports/eagle3-8b-scale-validation/analysis.json`
- `reports/d8-block{8,16,32}/benchmark-paper-summary.json`
- the four relay fitting summaries already loaded by `_load_training`
- the target-specific control summaries already loaded by `_load_native_controls`

### Task 1: Load and validate paired request latency distributions

**Files:**

- Modify: `tests/test_iclr_paper_assets.py`
- Modify: `scripts/build_iclr_paper_assets.py:40-121`

**Step 1: Write the failing test**

Add this test after `test_main_math_values_match_final_artifacts`:

```python
def test_request_latency_ratios_are_exactly_paired() -> None:
    builder = _load_builder()
    rows = {
        (row["family_key"], row["target_key"]): row
        for row in builder.load_paper_data(ROOT)["request_speed_distributions"]
    }

    assert set(rows) == {
        ("dflash", "8b"),
        ("dflash", "14b"),
        ("eagle3", "8b"),
        ("eagle3", "14b"),
    }
    assert all(row["paired_requests"] == 500 for row in rows.values())
    assert all(len(row["ratios"]) == 500 for row in rows.values())
    assert rows[("dflash", "8b")]["faster_requests"] == 500
    assert rows[("dflash", "14b")]["faster_requests"] == 494
    assert rows[("eagle3", "8b")]["faster_requests"] == 500
    assert rows[("eagle3", "14b")]["faster_requests"] == 448
    assert rows[("dflash", "8b")]["quantiles"] == pytest.approx(
        [1.3343116812342506, 1.494643024301611, 1.6539237696829243]
    )
    assert rows[("dflash", "14b")]["quantiles"] == pytest.approx(
        [1.0807169004026742, 1.2419410971894964, 1.370045598898449]
    )
    assert rows[("eagle3", "8b")]["quantiles"] == pytest.approx(
        [1.145528176462032, 1.2531476886276791, 1.3779952429998785]
    )
    assert rows[("eagle3", "14b")]["quantiles"] == pytest.approx(
        [0.9705087719556503, 1.078996154576983, 1.185377283671023]
    )
```

Also add `import pytest` at the top of the file.

**Step 2: Run the test and verify it fails**

Run:

```bash
.venv/bin/pytest tests/test_iclr_paper_assets.py::test_request_latency_ratios_are_exactly_paired -v
```

Expected: FAIL with `KeyError: 'request_speed_distributions'`.

**Step 3: Implement the JSONL reader and exact pairing**

Add `import numpy as np` beside the Matplotlib imports. Add these functions after `_read_json`:

```python
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_request_speed_distributions(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, target, directory, source_name, relay_name in _main_specifications():
        records: list[dict[str, Any]] = []
        for path in sorted((root / "reports" / "final" / directory).glob("benchmark-rank*.jsonl")):
            records.extend(_read_jsonl(path))

        by_method: dict[str, dict[tuple[str, str, int, int], float]] = {}
        for record in records:
            if record.get("benchmark") != "math500":
                continue
            method = str(record["method"])
            if method not in {source_name, relay_name}:
                continue
            key = (
                str(record["benchmark"]),
                str(record["problem_id"]),
                int(record["repetition"]),
                int(record.get("turn_index", 0)),
            )
            if key in by_method.setdefault(method, {}):
                raise ValueError(f"duplicate request key for {directory}/{method}: {key}")
            by_method[method][key] = float(record["request_seconds"])

        source = by_method[source_name]
        relay = by_method[relay_name]
        if source.keys() != relay.keys():
            missing_source = sorted(relay.keys() - source.keys())
            missing_relay = sorted(source.keys() - relay.keys())
            raise ValueError(
                f"unpaired requests in {directory}: "
                f"missing_source={missing_source[:3]}, missing_relay={missing_relay[:3]}"
            )
        keys = sorted(source)
        ratios = np.asarray([source[key] / relay[key] for key in keys], dtype=float)
        rows.append(
            {
                "family_key": family,
                "family": FAMILY_LABEL[family],
                "target_key": target,
                "target": TARGET_LABEL[target],
                "paired_requests": len(keys),
                "ratios": ratios.tolist(),
                "faster_requests": int(np.count_nonzero(ratios > 1.0)),
                "quantiles": np.quantile(ratios, [0.05, 0.50, 0.95]).tolist(),
            }
        )
    return rows
```

Add this entry to the dictionary returned by `load_paper_data`:

```python
"request_speed_distributions": _load_request_speed_distributions(root),
```

**Step 4: Run the focused and asset tests**

Run:

```bash
.venv/bin/pytest tests/test_iclr_paper_assets.py::test_request_latency_ratios_are_exactly_paired -v
.venv/bin/pytest tests/test_iclr_paper_assets.py -v
```

Expected: both commands PASS. If the pre-registered tolerances differ from regenerated values, inspect the exact pairing before changing them. Never round values in the loader.

**Step 5: Commit the loader**

```bash
git add scripts/build_iclr_paper_assets.py tests/test_iclr_paper_assets.py
git commit -m "feat: derive paired request latency evidence"
```

### Task 2: Load registered design-selection evidence

**Files:**

- Modify: `tests/test_iclr_paper_assets.py`
- Modify: `scripts/build_iclr_paper_assets.py:162-299`

**Step 1: Write the failing test**

Add:

```python
def test_design_checks_match_registered_artifacts() -> None:
    builder = _load_builder()
    checks = builder.load_paper_data(ROOT)["design_checks"]

    objective = checks["objective"]
    assert objective["8b"]["relative_speedup"] == pytest.approx(1.5226880222674086)
    assert objective["8b"]["historical_speedup"] == pytest.approx(1.5094910550909824)
    assert objective["14b"]["relative_over_historical"] == pytest.approx(
        1.0114135653214098
    )

    scale = checks["eagle_scale"]
    assert scale["normalized"]["acceptance_retention"] == pytest.approx(
        0.6903449176999565
    )
    assert scale["normalized"]["speedup"] == pytest.approx(1.0350501271374102)
    assert scale["preserved"]["acceptance_retention"] == pytest.approx(0.826, abs=5e-4)
    assert scale["preserved"]["speedup"] == pytest.approx(1.237, abs=5e-4)

    block = checks["dflash_block_length"]
    assert [row["block_length"] for row in block] == [8, 16, 32]
    assert [row["relay_tps"] for row in block] == pytest.approx(
        [164.8967985323898, 210.14740573133233, 131.36258373999416]
    )
```

**Step 2: Run the test and verify it fails**

```bash
.venv/bin/pytest tests/test_iclr_paper_assets.py::test_design_checks_match_registered_artifacts -v
```

Expected: FAIL with `KeyError: 'design_checks'`.

**Step 3: Implement the loader**

Add after `_load_heldout_forecasts`:

```python
def _load_design_checks(root: Path) -> dict[str, Any]:
    objective_raw = _read_json(
        root / "reports" / "design-selection" / "dflash" / "objective-selection.json"
    )["targets"]
    objective: dict[str, dict[str, float | list[float]]] = {}
    for target, model_name in (("8b", "Qwen/Qwen3-8B"), ("14b", "Qwen/Qwen3-14B")):
        row = objective_raw[model_name]
        objective[target] = {
            "relative_speedup": float(row["relative"]["source_relative_speedup"]),
            "historical_speedup": float(row["historical_0.1"]["source_relative_speedup"]),
            "relative_over_historical": float(row["relative_over_historical_speed"]),
            "relative_over_historical_ci": [
                float(value) for value in row["relative_over_historical_speed_95_percent"]
            ],
        }

    normalized = _read_json(
        root
        / "reports"
        / "design-selection"
        / "eagle3"
        / "normalized-linear-analysis.json"
    )
    preserved = _read_json(root / "reports" / "eagle3-8b-scale-validation" / "analysis.json")
    eagle_scale = {
        "normalized": {
            "acceptance_retention": float(normalized["acceptance_retention"]),
            "speedup": float(normalized["observed_speedup"]),
            "speedup_ci": [
                float(normalized["paired_bootstrap_95_percent"]["lower"]),
                float(normalized["paired_bootstrap_95_percent"]["upper"]),
            ],
        },
        "preserved": {
            "acceptance_retention": float(preserved["acceptance_retention"]),
            "speedup": float(preserved["observed_speedup"]),
            "speedup_ci": [
                float(preserved["paired_bootstrap_95_percent"]["lower"]),
                float(preserved["paired_bootstrap_95_percent"]["upper"]),
            ],
        },
    }

    block_rows: list[dict[str, float | int]] = []
    for length in (8, 16, 32):
        raw = _read_json(
            root / "reports" / f"d8-block{length}" / "benchmark-paper-summary.json"
        )["by_benchmark"]["math500"]["methods"]
        block_rows.append(
            {
                "block_length": length,
                "relay_tps": float(raw["relay_f"]["end_to_end_tokens_per_second"]),
                "source_tps": float(
                    raw["naive_source_reuse"]["end_to_end_tokens_per_second"]
                ),
                "native_tps": float(
                    raw["native_target_dflash"]["end_to_end_tokens_per_second"]
                ),
            }
        )
    return {
        "objective": objective,
        "eagle_scale": eagle_scale,
        "dflash_block_length": block_rows,
    }
```

If `reports/eagle3-8b-scale-validation/analysis.json` uses a nested key rather than the flat fields above, read the artifact and use the exact recorded path. Do not copy rounded values from the manuscript.

Add to `load_paper_data`:

```python
"design_checks": _load_design_checks(root),
```

**Step 4: Run the focused tests**

```bash
.venv/bin/pytest tests/test_iclr_paper_assets.py::test_design_checks_match_registered_artifacts -v
.venv/bin/pytest tests/test_iclr_paper_assets.py -v
```

Expected: PASS.

**Step 5: Commit the loader**

```bash
git add scripts/build_iclr_paper_assets.py tests/test_iclr_paper_assets.py
git commit -m "feat: load RelaySpec design evidence"
```

### Task 3: Add latency-model validation figure

**Files:**

- Modify: `tests/test_iclr_paper_assets.py`
- Modify: `scripts/build_iclr_paper_assets.py:522-1015`
- Generate: `paper/iclr2027/figures/latency_validation.pdf`

**Step 1: Make the reproducibility test expect the new figure**

Add `Path("figures/latency_validation.pdf")` to `expected_figures` in `test_generated_assets_are_reproducible`.

Add a semantic test:

```python
def test_latency_validation_contains_all_registered_cells() -> None:
    builder = _load_builder()
    data = builder.load_paper_data(ROOT)
    assert len(data["heldout_forecasts"]) == 4
    assert len(data["breadth_rows"]) == 16
    assert all(len(row["speedup_ci"]) == 2 for row in data["breadth_rows"])
    assert data["breadth_aggregate"]["amdahl_direction_matches"] == 16
```

**Step 2: Run the reproducibility test and verify it fails**

```bash
.venv/bin/pytest tests/test_iclr_paper_assets.py::test_generated_assets_are_reproducible -v
```

Expected: FAIL because `latency_validation.pdf` does not exist.

**Step 3: Implement `_draw_latency_validation`**

Create a 7.0 by 2.55 inch, two-panel figure.

- Panel A x-axis: `development_estimate`. y-axis: `heldout_speedup`.
- Draw vertical 95% confidence intervals from `heldout_ci`.
- Draw an identity line and label each point with proposer family and target size.
- State `mean absolute relative error = 1.45%` in the panel, sourced from `forecast_mean_absolute_relative_error_percent`.
- Panel B x-axis: `accounting_speedup`. y-axis: `speedup` for all 16 breadth rows.
- Draw vertical 95% confidence intervals from `speedup_ci`.
- Draw identity and speedup-one lines.
- Encode proposer with marker shape and task with fill. Use a short direct legend.
- Add panel titles `Held-out forecast` and `Same-run accounting check` so the two forms of evidence cannot be confused.
- Use `ax.set_aspect("equal", adjustable="box")` only if it does not make labels unreadable.

Add this call in `build_all`:

```python
_draw_latency_validation(data, figures / "latency_validation.pdf")
```

**Step 4: Regenerate and test**

```bash
.venv/bin/python scripts/build_iclr_paper_assets.py
.venv/bin/pytest tests/test_iclr_paper_assets.py -v
pdfinfo paper/iclr2027/figures/latency_validation.pdf
pdffonts paper/iclr2027/figures/latency_validation.pdf
```

Expected: tests PASS, PDF is one page, and every font row reports `emb=yes`.

**Step 5: Commit the figure**

```bash
git add scripts/build_iclr_paper_assets.py tests/test_iclr_paper_assets.py paper/iclr2027/figures/latency_validation.pdf
git commit -m "feat: visualize RelaySpec latency validation"
```

### Task 4: Expand the system diagram with the fitting path and measured stakes

**Files:**

- Modify: `tests/test_iclr_paper_assets.py`
- Modify: `scripts/build_iclr_paper_assets.py:574-614`
- Generate: `paper/iclr2027/figures/system_overview.pdf`

**Step 1: Add a source-level contract test**

Add:

```python
def test_system_overview_receives_measured_data() -> None:
    builder = _load_builder()
    assert builder._draw_system_overview.__code__.co_argcount == 2
```

**Step 2: Run it and verify it fails**

```bash
.venv/bin/pytest tests/test_iclr_paper_assets.py::test_system_overview_receives_measured_data -v
```

Expected: FAIL because the current function accepts only `path`.

**Step 3: Redesign `_draw_system_overview(data, path)`**

Use three aligned regions in one full-width figure:

1. `Source reuse during every cycle`: target verifier, committed hidden states, Qwen3-4B source trunk, proposer, target verification.
2. `Fit once for each target`: paired source context and target hidden states, relative interface MSE, fitted linear relay. Label the proposer and both language models as frozen.
3. `RelaySpec during every cycle`: target hidden states, relay, the same frozen proposer, unchanged target verification.

Add an evidence strip below the flow:

- DFlash source trunk share: 40.0% at 8B and 23.1% at 14B, derived from `main_math` rather than hard-coded.
- Relay computation share below 1% of RelaySpec request time, derive the displayed range from `relay_fraction_of_relay`.
- EAGLE-only peak memory saved at 8B and 14B, derive from `memory` and label the scope explicitly.

Use a dashed boundary around the offline fitting region. Use the same fill for the target verifier in both runtime panels. Use a hatch for the removed source trunk so grayscale preserves the comparison.

Change the `build_all` call to:

```python
_draw_system_overview(data, figures / "system_overview.pdf")
```

**Step 4: Regenerate and test**

```bash
.venv/bin/python scripts/build_iclr_paper_assets.py
.venv/bin/pytest tests/test_iclr_paper_assets.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/build_iclr_paper_assets.py tests/test_iclr_paper_assets.py paper/iclr2027/figures/system_overview.pdf
git commit -m "feat: show RelaySpec fitting and deployment flow"
```

### Task 5: Replace the headline plot with throughput plus request-level consistency

**Files:**

- Modify: `tests/test_iclr_paper_assets.py`
- Modify: `scripts/build_iclr_paper_assets.py:615-715`
- Generate: `paper/iclr2027/figures/main_result.pdf`
- Remove from generated output: `paper/iclr2027/figures/main_throughput.pdf`

**Step 1: Change the expected generated figure name**

Replace `Path("figures/main_throughput.pdf")` with `Path("figures/main_result.pdf")` in the expected figure set.

**Step 2: Run the test and verify it fails**

```bash
.venv/bin/pytest tests/test_iclr_paper_assets.py::test_generated_assets_are_reproducible -v
```

Expected: FAIL because `main_result.pdf` is absent.

**Step 3: Implement `_draw_main_result(data, path)`**

Make a two-panel 7.0 by 3.0 inch figure.

Panel A:

- Retain the existing absolute throughput markers for native autoregressive decode, optimized source reuse, RelaySpec, and the available target-specific proposer.
- Keep the line between source reuse and RelaySpec.
- Print each aggregate RelaySpec/source ratio and paired bootstrap interval above the line.
- Add a compact right-hand annotation for each row: `accuracy S/R`, then exact text count. Use exact values from `main_math`.
- Mark the missing DFlash-14B target-specific proposer as `not available` rather than plotting zero.

Panel B:

- For each `request_speed_distributions` row, sort its 500 ratios and plot `np.arange(1, n + 1) / n` against the ratio.
- Use step-style empirical cumulative distributions.
- Draw a vertical line at one.
- Direct-label each curve at its median. Include `faster_requests/500` in the label.
- Use the x-label `Paired request latency ratio, source reuse / RelaySpec`.
- State in the caption, not the plot, that this distribution is descriptive and that aggregate throughput remains the primary estimator.

Call it from `build_all` and stop generating `main_throughput.pdf`:

```python
_draw_main_result(data, figures / "main_result.pdf")
```

**Step 4: Regenerate and remove the obsolete asset**

```bash
.venv/bin/python scripts/build_iclr_paper_assets.py
git rm --ignore-unmatch paper/iclr2027/figures/main_throughput.pdf
.venv/bin/pytest tests/test_iclr_paper_assets.py -v
```

Expected: PASS. Confirm the four faster-request counts are 500, 494, 500, and 448.

**Step 5: Commit**

```bash
git add scripts/build_iclr_paper_assets.py tests/test_iclr_paper_assets.py paper/iclr2027/figures/main_result.pdf
git add -u paper/iclr2027/figures/main_throughput.pdf
git commit -m "feat: add request-level evidence to headline result"
```

### Task 6: Make the mechanism and break-even plots answer causal questions directly

**Files:**

- Modify: `scripts/build_iclr_paper_assets.py:716-918`
- Generate: `paper/iclr2027/figures/mechanism.pdf`
- Generate: `paper/iclr2027/figures/breadth_and_margin.pdf`
- Test: `tests/test_iclr_paper_assets.py`

**Step 1: Add a break-even classification test**

```python
def test_break_even_boundary_matches_measured_speed_direction() -> None:
    builder = _load_builder()
    rows = builder.load_paper_data(ROOT)["breadth_rows"]
    assert all(
        (row["acceptance_retention"] > row["break_even_retention"])
        == (row["accounting_speedup"] > 1.0)
        for row in rows
    )
```

**Step 2: Run the test**

```bash
.venv/bin/pytest tests/test_iclr_paper_assets.py::test_break_even_boundary_matches_measured_speed_direction -v
```

Expected: PASS. This is a regression test that protects the visual interpretation.

**Step 3: Improve `mechanism.pdf`**

- Keep normalized timing stacks for target, source trunk, proposer, and other runtime.
- Add direct labels to source-trunk segments, because this is the removable work.
- Keep ideal versus observed gain in the second panel.
- Show acceptance retention beside each observed point so the gap from ideal has a concrete explanation.
- Increase all tick and annotation text to at least 7.5 pt.
- Use hatches on stack segments and filled versus open markers on ideal/observed points.

**Step 4: Replace the margin panel in `breadth_and_margin.pdf`**

- Keep the 4 by 4 measured speedup heatmap.
- Use a solid cell border where the 95% interval is above one and a dashed cell border otherwise.
- Set the second panel x-axis to `break_even_retention` and y-axis to `acceptance_retention`.
- Draw the diagonal `y=x` and shade the above-diagonal region lightly as `predicted faster`.
- Encode proposer family by marker and target size by fill.
- Label the three cells closest to the boundary and any slower cell far from it. Do not label all 16 points.

**Step 5: Regenerate and test**

```bash
.venv/bin/python scripts/build_iclr_paper_assets.py
.venv/bin/pytest tests/test_iclr_paper_assets.py -v
```

Expected: PASS.

**Step 6: Commit**

```bash
git add scripts/build_iclr_paper_assets.py tests/test_iclr_paper_assets.py paper/iclr2027/figures/mechanism.pdf paper/iclr2027/figures/breadth_and_margin.pdf
git commit -m "feat: clarify RelaySpec mechanism and break-even evidence"
```

### Task 7: Consolidate fitting, memory, and design evidence into one figure

**Files:**

- Modify: `tests/test_iclr_paper_assets.py`
- Modify: `scripts/build_iclr_paper_assets.py:919-1015`
- Generate: `paper/iclr2027/figures/resource_and_design.pdf`
- Stop generating in the main asset set: `paper/iclr2027/figures/memory_reduction.pdf`

**Step 1: Update expected assets**

Replace `Path("figures/memory_reduction.pdf")` with `Path("figures/resource_and_design.pdf")` in `expected_figures`.

**Step 2: Run the test and verify it fails**

```bash
.venv/bin/pytest tests/test_iclr_paper_assets.py::test_generated_assets_are_reproducible -v
```

Expected: FAIL because the consolidated figure is absent.

**Step 3: Implement `_draw_resource_and_design(data, path)`**

Use a compact 2 by 3 layout no taller than 3.4 inches. Five panels contain evidence and the sixth cell contains a concise legend plus scope notes.

- Panel A, `Fit once`: four bars for DFlash/EAGLE-3 at 8B/14B using `training_seconds`. Print seconds directly.
- Panel B, `Native proposer throughput retained`: plot RelaySpec throughput divided by each available target-specific proposer throughput. Omit DFlash-14B and label it unavailable.
- Panel C, `EAGLE peak memory`: grouped source-reuse and RelaySpec bars for 8B and 14B. Print GiB saved. Label this panel EAGLE-only.
- Panel D, `DFlash fitting loss`: paired points for relative MSE and historical MSE plus 0.1 cosine at 8B and 14B. Plot the direct relative-over-historical ratio and its interval on a secondary annotation, not a second y-axis.
- Panel E, `EAGLE input scale`: two paired points for normalized and preserved magnitude. Use speedup as the y-axis and print acceptance retention beside each point.
- Panel F, `DFlash block length`: line plot of 8, 16, and 32 against RelaySpec tokens per second. Mark 16 as the registered selection.

Do not put incompatible units on the same axis. Every panel gets its own y-axis label or direct unit label. Use no more than one small legend in the sixth cell.

Change `build_all` to call:

```python
_draw_resource_and_design(data, figures / "resource_and_design.pdf")
```

Stop calling `_draw_memory` from `build_all`. Keep `_draw_memory` only if the appendix still uses it. Otherwise remove the function after the manuscript no longer references its file.

**Step 4: Regenerate and test**

```bash
.venv/bin/python scripts/build_iclr_paper_assets.py
.venv/bin/pytest tests/test_iclr_paper_assets.py -v
pdfinfo paper/iclr2027/figures/resource_and_design.pdf
```

Expected: PASS and one-page vector PDF.

**Step 5: Commit**

```bash
git add scripts/build_iclr_paper_assets.py tests/test_iclr_paper_assets.py paper/iclr2027/figures/resource_and_design.pdf
git commit -m "feat: consolidate RelaySpec resource and design evidence"
```

### Task 8: Compress the main manuscript around the new figures

**Files:**

- Modify: `paper/iclr2027/relayspec_iclr2027.tex:49-600`
- Modify: `tests/test_iclr_manuscript_audit.py`
- Reuse: `paper/iclr2027/generated/main_math_table.tex`
- Reuse: `paper/iclr2027/generated/resource_table.tex`
- Reuse: `paper/iclr2027/generated/forecast_table.tex`

**Step 1: Add main-figure and appendix-table assertions**

Add to `test_paper_is_anonymous_and_uses_generated_results`:

```python
assert "figures/latency_validation.pdf" in source
assert "figures/main_result.pdf" in source
assert "figures/resource_and_design.pdf" in source
assert source.index("generated/main_math_table.tex") > source.index("\\appendix")
assert source.index("generated/resource_table.tex") > source.index("\\appendix")
```

**Step 2: Run and verify the test fails**

```bash
.venv/bin/pytest tests/test_iclr_manuscript_audit.py::test_paper_is_anonymous_and_uses_generated_results -v
```

Expected: FAIL because the new figures are not yet referenced and the two tables remain in the main body.

**Step 3: Rewrite the main body to the approved page sequence**

Edit only normal prose and floats. Keep the title anonymous and do not add a conference status banner.

Page 1:

- Keep the abstract under approximately 170 words.
- Reduce the introduction to four paragraphs: deployment problem, why source reuse is costly, method plus correctness boundary, measured contributions.
- State the gap once: existing target-trained feature-conditioned proposers cannot directly consume hidden states from a changed target, so deployment either retains the source transformer or retrains a proposer.

Page 2:

- Insert `figures/system_overview.pdf` at `\linewidth`.
- Merge the deployment use case into one paragraph below the figure.
- Keep compatibility criteria as a short numbered list or compact table.

Page 3:

- Preserve Equations 1 to 4 and the interface table.
- Remove repeated descriptions of the same five hidden states and dimensions. Give tensor dimensions once in the method and full dimensions in Appendix `app:method`.

Page 4:

- Preserve all six inference steps.
- Preserve the correctness proposition and compress its proof sketch to one paragraph.
- Preserve Equations 5 to 7 and the break-even condition.

Page 5:

- Insert `figures/latency_validation.pdf`.
- Caption Panel A as a forecast from the 32 registered development prompts to the untouched 468-prompt MATH subset.
- Caption Panel B as a same-run accounting check, not independent prediction.
- Keep the experimental setup in two compact paragraphs with model IDs, fitting record count, task counts, four RTX 6000 Ada GPUs, BF16, greedy non-thinking decode, batch one, synchronized end-to-end timing, output-token throughput, and paired cluster bootstrap.
- Move optimizer, warm-up, backend, and exact manifest details to Appendix `app:protocol`.

Page 6:

- Insert `figures/main_result.pdf`.
- Remove the complete four-row main result table from the main body.
- Keep one results paragraph that reports all four aggregate speedups with intervals, accuracy equality, and exact-text counts.
- State that request-level curves describe consistency but do not replace aggregate throughput.

Page 7:

- Place `figures/mechanism.pdf` and `figures/breadth_and_margin.pdf` as two stacked full-width figures or one `figure*` with two rows.
- Keep one paragraph for removed source work and one for breadth plus break-even behavior.
- Do not repeat values already printed in the panels unless they are central claims.

Page 8:

- Insert `figures/resource_and_design.pdf`.
- Remove the complete resource table from the main body.
- Replace the current `Checks that determined the final design` subsection with one compact paragraph tied to the panels.
- Compress related work into two paragraphs: proposer improvement and portable/cache/interface reuse. Preserve the novelty boundary and all citations.

Page 9:

- Preserve limitations, conclusion, AI use, ethics, and reproducibility statements.
- Keep `\label{maintextend}` after limitations and conclusion and before required unnumbered statements if that remains consistent with the existing audit. Do not hide substantive method or result content after this label.
- Allow references to start on page 9.

Use captions to carry procedure and scope that a reader needs to interpret the plot. Do not use captions to make new claims.

**Step 4: Move complete tables to the appendix**

Under `\section{Full workload results}` or a new `\subsection{Complete MATH headline results}`, insert:

```latex
\begin{table}[H]
\caption{Complete MATH-500 throughput, task accuracy, and exact-text results. Throughput is total emitted tokens divided by total synchronized request time. Confidence intervals use paired request-cluster bootstrap.}
\label{tab:main-results}
\centering
\small
\input{generated/main_math_table.tex}
\end{table}
```

Under the resource or ablation appendix, insert:

```latex
\begin{table}[H]
\caption{Relay fitting cost, throughput retained relative to available target-specific proposers, and isolated EAGLE memory savings. D/E denotes DFlash and EAGLE-3. The DFlash-14B target-specific proposer was unavailable.}
\label{tab:resources}
\centering
\small
\input{generated/resource_table.tex}
\end{table}
```

Ensure each label occurs exactly once.

**Step 5: Run source-level tests**

```bash
.venv/bin/pytest tests/test_iclr_manuscript_audit.py -v
```

Expected: source-level tests PASS. The full submission audit can still fail until the PDF is rebuilt.

**Step 6: Commit the manuscript rewrite**

```bash
git add paper/iclr2027/relayspec_iclr2027.tex tests/test_iclr_manuscript_audit.py
git commit -m "docs: restructure RelaySpec paper around visual evidence"
```

### Task 9: Build the paper and stabilize float placement within nine pages

**Files:**

- Modify as needed: `paper/iclr2027/relayspec_iclr2027.tex`
- Generate: `paper/iclr2027/relayspec_iclr2027.pdf`
- Generate: `output/pdf/RelaySpec_ICLR_2027.pdf`

**Step 1: Build generated assets**

```bash
.venv/bin/python scripts/build_iclr_paper_assets.py
```

Expected: `Generated 4 main cells and 16 breadth cells`.

**Step 2: Compile through all bibliography passes**

Run from `paper/iclr2027`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error relayspec_iclr2027.tex
bibtex relayspec_iclr2027
pdflatex -interaction=nonstopmode -halt-on-error relayspec_iclr2027.tex
pdflatex -interaction=nonstopmode -halt-on-error relayspec_iclr2027.tex
```

Expected: all commands exit zero.

**Step 3: Inspect build warnings and page boundary**

```bash
rg -n "Overfull|Underfull|undefined|multiply defined|LaTeX Warning" relayspec_iclr2027.log
rg -n "newlabel\{maintextend\}" relayspec_iclr2027.aux
pdfinfo relayspec_iclr2027.pdf | rg "Pages|Page size"
```

Expected:

- no overfull boxes,
- no undefined citations or references,
- `maintextend` on page 9 or earlier,
- US Letter page size.

**Step 4: Fix float order using only legitimate layout changes**

If the boundary exceeds page 9, make changes in this order:

1. remove repeated prose already stated in captions or panels,
2. shorten captions without removing estimator, sample, or scope information,
3. move complete numerical details to existing appendix tables,
4. resize an individual figure only while plot text stays readable,
5. use standard `[t]`, `[tb]`, or `figure*` placement.

Do not alter margins, global fonts, line spacing, or use negative `\vspace`.

Recompile after each change. Stop once the main boundary is at page 9 or earlier and the sequence remains coherent.

**Step 5: Copy the verified draft to the delivery path**

```bash
mkdir -p output/pdf
cp paper/iclr2027/relayspec_iclr2027.pdf output/pdf/RelaySpec_ICLR_2027.pdf
```

This copy is allowed because it duplicates an already generated build product. It does not edit source.

**Step 6: Commit source and generated PDF**

```bash
git add paper/iclr2027/relayspec_iclr2027.tex paper/iclr2027/relayspec_iclr2027.pdf output/pdf/RelaySpec_ICLR_2027.pdf paper/iclr2027/generated paper/iclr2027/figures
git commit -m "docs: build dense official-format RelaySpec paper"
```

### Task 10: Render and inspect every page in color and grayscale

**Files:**

- Create or update: `tmp/iclr-review/color/page-*.png`
- Create or update: `tmp/iclr-review/grayscale/page-*.png`
- Modify: `reports/ICLR_VISUAL_REVIEW.json`

**Step 1: Render every PDF page in color**

```bash
rm -rf tmp/iclr-review/color tmp/iclr-review/grayscale
mkdir -p tmp/iclr-review/color tmp/iclr-review/grayscale
pdftoppm -png -r 150 output/pdf/RelaySpec_ICLR_2027.pdf tmp/iclr-review/color/page
```

Expected: one PNG per PDF page.

**Step 2: Render grayscale copies**

```bash
for image in tmp/iclr-review/color/page-*.png; do
  name=$(basename "$image")
  convert "$image" -colorspace Gray "tmp/iclr-review/grayscale/$name"
done
```

Expected: matching grayscale page count.

**Step 3: Inspect the pages**

Use `view_image` on every color page and every grayscale page. Check:

- figure order follows the approved six-question sequence,
- no plot, caption, equation, or table is clipped,
- no text or float overlaps,
- panel lettering and direct labels are readable at normal zoom,
- color-coded meaning remains clear in grayscale,
- figures do not leave large unexplained blank regions,
- tables begin in the appendix,
- references and appendix transition are not mistaken for main result content,
- no status line says the paper is under review.

Do not mark this step complete from a contact sheet alone. Contact sheets may be used for navigation, but every page must be opened at readable resolution.

**Step 4: Fix and rerender until all checks pass**

For each visual defect, edit the smallest relevant plot or LaTeX float, rebuild the asset and PDF, rerender the affected pages, and inspect them again. After any page-count change, rerender all pages.

**Step 5: Update the visual-review record**

Calculate the PDF digest:

```bash
sha256sum output/pdf/RelaySpec_ICLR_2027.pdf
```

Update `reports/ICLR_VISUAL_REVIEW.json` with:

- the exact SHA-256,
- page count,
- review date `2026-08-28`,
- color and grayscale review set paths,
- explicit booleans for clipping, overlap, float order, legibility, grayscale meaning, appendix transition, and status-line absence,
- a short note that every page was inspected at 150 DPI.

Use `apply_patch` for the JSON edit.

**Step 6: Commit the reviewed output**

```bash
git add reports/ICLR_VISUAL_REVIEW.json paper/iclr2027/relayspec_iclr2027.tex paper/iclr2027/relayspec_iclr2027.pdf output/pdf/RelaySpec_ICLR_2027.pdf paper/iclr2027/figures
git commit -m "docs: record RelaySpec manuscript visual review"
```

Do not add `tmp/iclr-review` unless it is already part of the repository policy.

### Task 11: Run complete scientific and submission verification

**Files:**

- Modify only if a check reveals a real defect: paper sources, asset builder, or tests
- Final delivery: `output/pdf/RelaySpec_ICLR_2027.pdf`

**Step 1: Run the full automated test suite**

```bash
.venv/bin/pytest -q
```

Expected: all tests PASS.

**Step 2: Run lint**

```bash
.venv/bin/ruff check .
```

Expected: `All checks passed!`

**Step 3: Run the manuscript audit**

```bash
.venv/bin/python scripts/audit_iclr_manuscript.py \
  --root . \
  --paper paper/iclr2027 \
  --pdf output/pdf/RelaySpec_ICLR_2027.pdf
```

Expected: every audit check passes, including official style hashes, generated-asset reproducibility, main page boundary, anonymity, citation resolution, PDF parsing, embedded fonts, page size, and visual-review digest.

**Step 4: Perform final PDF checks**

```bash
pdfinfo output/pdf/RelaySpec_ICLR_2027.pdf
pdffonts output/pdf/RelaySpec_ICLR_2027.pdf
pdftotext output/pdf/RelaySpec_ICLR_2027.pdf tmp/relayspec-final.txt
rg -n "Under review as a conference paper|undefined|TODO|TBD" tmp/relayspec-final.txt
gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=nullpage output/pdf/RelaySpec_ICLR_2027.pdf
```

Expected:

- US Letter page size,
- all fonts embedded,
- no banned status phrase or unresolved placeholder,
- Ghostscript exit code zero.

**Step 5: Check the final diff for scope and accidental data changes**

```bash
git status --short
git diff --stat HEAD~1..HEAD
git diff --check
```

Expected: no whitespace errors. Confirm no unrelated dirty file was staged or overwritten.

**Step 6: If verification required changes, make one final focused commit**

```bash
git add <only-the-files-fixed>
git commit -m "fix: complete RelaySpec submission verification"
```

Do not create an empty commit when all checks already pass.

## Completion criteria

The implementation is complete only when all of the following are true:

- the main body ends on page 9 or earlier under the unmodified official ICLR style,
- six main figures answer the approved evidence sequence and contain about thirteen readable panels,
- the complete MATH and resource tables appear in the appendix rather than being duplicated in the main body,
- all per-request ratios are paired by benchmark, problem, repetition, and turn,
- held-out forecast evidence is clearly separated from same-run accounting evidence,
- aggregate throughput remains the primary speed estimator,
- all plotted values reproduce from registered JSON or JSONL artifacts,
- accuracy, exact-text behavior, fitting cost, memory scope, and unavailable controls are stated without omission,
- every page passes color and grayscale visual inspection,
- the complete tests, linter, manuscript audit, font checks, parser checks, and placeholder search pass,
- the final PDF is available at `output/pdf/RelaySpec_ICLR_2027.pdf`.
