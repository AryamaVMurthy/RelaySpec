# Rendered appendix review: pages 36–62

Reviewed PDF: `paper/iclr2027/relayspec_iclr2027.pdf`.

SHA-256 of reviewed candidate:

`a1a670f9b661fb51cb75dd90ceec3724f5b33f7206abdc85ae20f530728a184d`

The hash was checked before and after inspection. Review date: 8 September 2026.

## Actual rendered-image coverage

All pages **36 through 62 inclusive** were reviewed in color and grayscale using `tmp/pdfs/visuals-final/{color,gray}/page-XX.png`.

- Three nine-page contact sheets per color mode were opened to check complete page sequence, overall density, ordering and whitespace.
- Every page in this range was additionally opened individually in color at its rendered resolution.
- Every figure page and dense-table page was additionally opened individually in grayscale: **36–57 and 59–62**. Page 58, consisting of prose and one equation, was inspected in the grayscale contact sheet and individually in color.
- Detailed new-figure inspections cover Figure 18 (task complexity), Figure 24 (selective-capture memory) and Figures 25–27 (verification work, request-level speed changes and empirical length distributions).
- This reviewer did not certify pages 1–35 in this round; those are assigned to the other reviewers.

## Results

No clipped figure, cut-off axis/legend, caption overlap, detached caption, broken table rule, missing panel, overprinted body text or improper float ordering was found in the reviewed range. Captions remain adjacent to their figures. The larger whitespace on some appendix pages is acceptable and avoids squeezing figures or captions.

One presentation issue was reported to the lead immediately: **Figures 19 and 20 on pages 42 and 43 have very small legacy axis ticks and eight-entry legends at their final placement.** The curves and captions remain interpretable, but the legend text benefits from zoom. The transfer agent is enlarging those source fonts while preserving the figure canvas/aspect ratio. These two pages require reinspection after rebuilding. This report is a review of the hash above, not an approval of a later PDF.

## Scientific and grayscale checks

| Pages | Inspected content | Finding |
|---|---|---|
| 36–38 | Regularization and EAGLE fitting trajectories, normalization/objective controls, validation-minimum decoding | Validation axes are explicitly separate from speed. Circle/square/triangle and solid/dashed encodings remain distinguishable in grayscale. Positive penalty changes are correctly labeled as worse validation. |
| 39–41 | Full-output small-data/capacity tables and subgroup plot | Same 128 exposed MATH questions and 2,048-token cap remain explicit. Complexity plot's six representative fits are distinguished from all twelve recorded methods. Dense N=2,048 is the denominator; group counts are legible and n=4 is shaded. Descriptive intervals and noncausal scope remain in the caption/prose. |
| 42–44 | 14B trajectories, rate checks and full-answer follow-up | No numerical/scope contradiction found. Sixteen-question capped screens remain distinct from the 32-GSM8K follow-up. Font issue in Figures 19/20 noted above. |
| 44–48 | Warm budgets and public-runtime alternatives | Tables fit their page width. Matched budgets and separate public runtimes are explicit. PARD/SD² exact-token disagreement is not hidden behind correct-answer counts. Selection and limited quality scope remain visible. |
| 49–50 | Acceptance-survival and sampled decoding | Survival is a fraction of recorded cycles, not draft-token acceptance. Solid/dashed source/relay curves remain legible without color. Sampled scores retain clustered two-seed/question accounting and are not treated as distributional proof. |
| 51–54 | Composition and frozen-checkpoint confirmation | Composition plot preserves both math losses and code/chat gains; marker shapes identify all four comparisons in grayscale. Throughput and task quality remain separate. Frozen speed criteria pass while one-point accuracy noninferiority remains unresolved. |
| 55–57 | Isolated source-removal and selective-capture memory | RTX 6000 Ada and L40S diagnostics remain distinct. Absolute memory and memory saved have appropriate units; selected/all-state line styles, tap marker shapes and open-ring lifetime controls survive grayscale. Caption preserves one synthetic prompt per length and excludes quality or material speed claims. |
| 57–59 | Rollout calibration and family protocol | Separate runtime, fitting costs, Numina split, original/selected recipes, exact AR reference, output cap and fitting-selection history remain clear. The richness of the rollout recipe is not attributed to the loss alone. |
| 60–61 | New verification and request/length graphics | Final-target tokens per verification-loop call are correctly labeled effective progress, not acceptance. Prefill calls are excluded and first-token/EOS/cap counting caveats remain explicit. Scatter diagonal is equality, not a fitted model. Length-bin counts, censoring at 2,048 and empirical rather than uncapped distributions remain explicit. Grayscale markers/line styles retain interpretability. |
| 62 | Inherited-training context | Native retention and original recipe record counts stay separate. Log scale is labeled, exact bar values are printed, and caption/prose explicitly disallow equivalent training-compute claims. |

## Required follow-up

Reinspect the final rebuilt versions of pages 42–43 after the announced legend/font correction, plus any other pages whose rendered pixels change. Remaining pages in this reviewed candidate pass the visual and scope checks above. The final paper's numerical evidence audit and other reviewers' page ranges remain separate verification steps.

## Final correction check — PASS

Final PDF SHA-256:

`123ab1d05d2e8bff1a46dc68581c108ddb5060ab3fb43a19b013082425be8348`

Reopened the final color and grayscale renderings of **pages 42 and 43** individually from `tmp/pdfs/visuals-final-r2/`. The enlarged axis labels, ticks and eight-entry legends are now readable at manuscript placement. Marker shapes, fill differences and dotted/dashed/solid styles preserve the eight trajectories in grayscale. Figure titles, panel titles and legends fit without clipping or overlap, both captions remain attached, and neighboring text/tables retain their layout. The one presentation issue from the first round is resolved.

Inspected `final-render-comparison.json`, which records that only pages 34, 42 and 43 changed in either color mode. Page 34 lies outside this reviewer's assigned scope. The remaining previously reviewed pages 36–62 are pixel-identical to the first candidate, so their earlier checks carry forward.

**Final status for pages 36–62: PASS**, covering the complete assigned range in color and grayscale, with no unresolved visual or caption/scope issue. This signoff is tied to the final hash above.
