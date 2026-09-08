# Appendix diagnostics implemented

Builder: `scripts/build_diagnostics_visuals.py`.

```bash
/home/aryamavmurthy/work/RelaySpec/.venv/bin/python scripts/build_diagnostics_visuals.py --root . --output paper/iclr2027
```

`build(root, output)` also supports import by the manuscript audit. Every figure is 6.2 inches wide with an 8.5–9.5 point plotting style. PDF and PNG files have deterministic metadata. The registry is `paper/iclr2027/generated/diagnostics_visuals_evidence.json`; it stores all consumed source hashes, exact plotted values, original subgroup evidence and scope notes.

## Figures and suggested placement/captions

### `visual_diagnostics_complexity.pdf`

Place in “Input-only task complexity,” after the explanatory paragraph or difficulty table.

Suggested caption:

> Input-only subgroup analysis of the same 128 MATH development questions. Six original 8,192-update fits show the small-data and capacity contrasts; throughput is relative to dense N=2,048 within each group. N is the fitting-record count, and linear/MLP numbers are hidden widths. Points show paired 95% request-bootstrap intervals. Labels give subgroup counts; the four-request input group is shaded to emphasize its limited support. Groups are observational, intervals do not adjust for multiple comparisons, and the complete twelve-method and subject analyses remain in the recorded evidence.

This makes the near-constant small-data retention over difficulty visible while showing the weakly supported long-input decline. Lines do not connect category means. Methods were selected as representative fixed-update contrasts, not from favorable subgroup outcomes. The two continuation fits, AR/native/source controls and dense reference remain in the complete table/source registry.

### `visual_diagnostics_quality.pdf`

Place in “Acceptance and numerical agreement,” or immediately after the primary paired quality table in the appendix.

Suggested caption:

> Quality and token agreement measure different outcomes in the primary BF16 Qwen suite. Left: RelaySpec minus matched AR MATH-500 accuracy in percentage points, with paired 95% request-bootstrap intervals. Right: complete token-sequence agreement counts on those same 500 requests. Every output cap is 2,048 tokens. These historical BF16/SDPA measurements are distinct from the exact batch-invariant BF16 rollout and FP32-target family extensions.

All four primary cells remain visible. Accuracy intervals cover zero without establishing equality or noninferiority. Exact-agreement bars use a zero-to-100% axis and do not reuse later 128-question results.

### `visual_diagnostics_memory.pdf`

Place in “Selective feature capture during generation,” next to the full capture-memory table.

Suggested caption:

> Selective feature capture reduces allocated memory as input length grows on DFlash-8B/L40S. Left: peak allocated memory when returning all states or retaining only the selected two/five taps. Right: memory saved, with open rings for the release-prefill lifetime control. Each length uses one synthetic archive prompt and two reversed-order executions at a 64-token output cap; the repetitions are not independent prompt samples. All compared outputs and acceptance trajectories match. This measures allocation rather than long-context quality or a material speed improvement.

The release-prefill control points overlap the original savings, making the controlled result explicit. This is separate from source-transformer-removal memory on RTX 6000 Ada. Both resident mapper copies remain in absolute peak measurements.

### `visual_diagnostics_native.pdf`

Place in the native interface compression discussion after both DFlash and EAGLE frozen-confirmation tables, or before “Code benefits from an additional late layer.”

Suggested caption:

> Native Qwen3-8B projection compression on separate frozen 64-question GSM8K sets. Throughput is relative to each released native drafter; whiskers are paired 95% request-bootstrap intervals at a 2,048-token output cap. Labels count only projection parameters. DFlash compares one/two-layer students with repacking and rank truncation. EAGLE-3 compares inherited two-layer initialization after 128 or 8,192 updates with the same types of controls. The dotted line is the 95% retention criterion. These comparisons use different frozen question sets and do not equate fitting budgets between families.

All five arms from each confirmation are plotted, including the one-layer DFlash loss and the EAGLE rank-truncation loss. Parameter counts follow k×4096×4096 for k dense taps and r×(20480+4096) for rank r. The graph does not imply that slicing columns removes the need for calibration.

## Completed checks

- Four vector figures and four PNG previews built successfully.
- 191 source files consumed and hash checked against their registries where a recorded hash exists.
- Raw per-method request coverage and token/time aggregate throughput checked for task subgroups, primary quality runs and native compression runs.
- Primary quality figures additionally match the scorer's output hashes to timed outputs and recompute correct fractions and accuracy differences.
- Primary full-sequence match counts recomputed from raw output hashes and output lengths.
- Memory points checked against raw allocated bytes, saved bytes and exact paired outputs/acceptance/target/draft calls/checkpoint hashes.
- A second build in a temporary output directory reproduced the registry exactly and all eight PDF/PNG assets byte for byte.
- All four PNG previews visually inspected; native-panel title/x-axis clipping found and corrected before the final deterministic check.

The images have been reviewed as standalone figures. Their final manuscript placement, scaling and captions still require the final rendered-paper review.
