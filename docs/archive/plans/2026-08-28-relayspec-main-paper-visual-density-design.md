# RelaySpec Main-Paper Visual Density Design

## Objective

Increase the amount of useful visual evidence in the RelaySpec main paper while preserving the official ICLR style and the nine-page main-paper limit. The redesign must keep the method, assumptions, equations, experimental protocol, limitations, and required statements scientifically complete.

## Chosen approach

Use a balanced evidence-first layout with six main figures containing about thirteen panels. Compress repeated prose and move complete numerical tables and procedural detail to the appendix. Do not change margins, global font size, line spacing, or the official style file.

The visual sequence answers six questions in order:

1. What redundant computation exists?
2. What does RelaySpec replace?
3. Does the latency model predict measured behavior?
4. How much faster is the resulting system?
5. Why does it speed up some configurations and slow down others?
6. What fitting, memory, and design costs remain?

## Main figures

### Figure 1: Deployment problem and RelaySpec

Expand the current source-reuse versus RelaySpec diagram with an offline fitting inset. Show that one frozen proposer receives a separate fitted map for each target. Add a compact evidence strip with the measured source-time share, relay-time share, and EAGLE memory reduction. Label the memory result as EAGLE-only.

### Figure 2: Latency-model validation

Create two panels.

- Panel A plots speedup estimated on 32 development prompts against speedup measured on the untouched 468 MATH prompts. It contains four proposer-target pairs, confidence intervals, and an identity line.
- Panel B plots Equation 6 predicted speedup against measured speedup for all 16 breadth cells. It contains measured confidence intervals, an identity line, and reference lines at one.

Keep the distinction between held-out forecasting and same-run accounting explicit in the caption.

### Figure 3: MATH throughput and request-level consistency

Create two panels.

- Panel A retains absolute throughput markers for autoregressive decoding, optimized source reuse, RelaySpec, and target-specific proposers. Print aggregate speedup intervals, task accuracy, and exact-text counts next to each row.
- Panel B shows empirical cumulative distributions of paired per-request latency ratios. Mark a ratio of one and report how many of 500 requests are faster for each configuration.

The aggregate throughput estimator remains the primary result. The request-level distribution is descriptive and must not replace the paired aggregate estimator.

### Figure 4: Removed work and retained gain

Retain the normalized source-reuse timing stacks and the comparison between ideal and measured gain. Directly label removable source share and accepted-token retention. Increase plot text size and use marker shapes and fill in addition to color.

### Figure 5: Workload breadth and break-even boundary

Retain the four-by-four throughput heatmap. Replace the current margin scatter with a direct break-even plot whose x-axis is required acceptance retention and whose y-axis is measured acceptance retention. Draw the diagonal boundary and visually distinguish faster and slower regions. Use line styles as well as color for confidence-interval status in the heatmap.

### Figure 6: Resource cost and design evidence

Create a compact multi-panel figure covering:

- fitting time and throughput recovery relative to released target-specific proposers,
- EAGLE source-reuse versus RelaySpec peak memory,
- relative MSE versus the historical MSE-plus-cosine objective,
- scale-preserving versus normalized EAGLE input,
- DFlash block lengths 8, 16, and 32.

The full numerical resource and design-check tables remain in the appendix.

## Text compression

- Reduce the introduction from six paragraphs to four and state the research gap once.
- Combine repeated speculative-cycle and compatibility explanations.
- Reduce the deployment-use-case section to one paragraph because Figure 1 carries its measured evidence.
- Preserve Equations 1 to 4, the interface table, and the six inference steps. Remove repeated tensor descriptions.
- Preserve the correctness proposition and compress its proof sketch to one paragraph.
- Preserve Equations 5 to 7. Replace forecast and accounting prose with Figure 2.
- Keep model names, fitting records, task counts, scoring, hardware, precision, decoding mode, timing unit, and uncertainty method in the main setup. Move optimizer, warm-up, backend, and complete configuration detail to the appendix.
- Move the complete MATH result table and resource table to the appendix. Embed their headline values in Figures 3 and 6.
- Reduce result interpretation to one paragraph per scientific question.
- Compress related work to two paragraphs without dropping citations or novelty boundaries.
- Preserve limitations and required AI, ethics, and reproducibility statements.

## Target page layout

| Page | Content |
|---:|---|
| 1 | Abstract and compressed introduction |
| 2 | Figure 1, compatibility, and deployment case |
| 3 | RelaySpec formulation and interface table |
| 4 | Inference algorithm, correctness, and break-even equations |
| 5 | Figure 2 and compact experimental setup |
| 6 | Figure 3 and headline results |
| 7 | Figures 4 and 5 |
| 8 | Figure 6 and related work |
| 9 | Limitations, required statements, and start of references |

## Content moved to the appendix

- Complete four-row MATH result table.
- Complete resource table.
- Full 16-cell workload table.
- Full forecast and latency-accounting tables.
- Complete fitting and runtime configuration.
- Complete objective, input-scale, and block-length tables.
- Acceptance survival, mismatch decomposition, prompt audits, and memory procedure.

## Presentation constraints

- Keep the official ICLR style file unchanged.
- Do not alter margins, global font size, line spacing, or paper size.
- Do not use negative-spacing or float-placement hacks to evade the page limit.
- Keep plot text at approximately 7.5 points or larger in the final paper.
- Use color only as a redundant cue. Shapes, fills, hatches, and line styles must preserve meaning in grayscale.
- Use direct labels where possible and avoid repeated legends.
- Limit each row to at most three compact panels.
- Do not create plots from incomparable literature speedups, zero-valued audits, internal identifiers, or unavailable controls.

## Verification

- Add tests for every new derived quantity and generated figure.
- Verify per-request pairing and ECDF counts against the raw request logs.
- Verify all plotted values against final JSON artifacts.
- Compile the manuscript until the main-text boundary is no later than page 9.
- Run citation, font, PDF parser, page-size, and official-style checks.
- Render every page at 150 DPI in color and grayscale.
- Inspect float order, clipping, overlap, plot text, captions, and appendix transitions.
- Run the full test suite, linter, and manuscript audit before delivery.
