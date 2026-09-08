# Independent review of overview, budget, and block figures

Reviewer: scaling-visuals agent. Inspected `scripts/build_results_visuals.py`, all three generated PNGs, `ar_evidence.json`, `timed-budget-decoding.json`, and the plotted evidence. A standalone build into a temporary directory succeeded and verified 418 stored input hashes.

## Required fixes identified

1. **Budget throughput label:** the budget panel (a) uses `tokens_per_second` from the matched-budget analysis. For every method checked, this equals `16 × mean_output_tokens / request_seconds` exactly, so it is **end-to-end throughput including prefill**, not decode-only throughput. Change the y-axis from “Decoding tokens/s” to “End-to-end tokens/s” (or “End-to-end throughput (tokens/s)”). The evidence field `decoding_tps` should preferably be renamed for the same reason. The paired ratio panel uses the same consistent timing denominator and is numerically correct.
2. **Block-size legend overlap:** panel (a)'s lower-left legend overlaps the naive source-reuse point and line at block 8. Move the legend to the upper-left vacant area; the native curve's highest point is at block 16, so an upper-left legend can avoid the data with suitable short labels.
3. **Block-size y-label clipping:** panel (b)'s long rotated label “Recorded target-token progress / cycle” is visibly clipped at the top of the PNG. Shorten to “Recorded progress / cycle” and explain target-token accounting in the caption.

The root reported and addressed an earlier main-panel legend overlap. In the PNG inspected here, the main right legend already has sufficient space above the first row and does not obscure data.

## Scientific checks passed

- All four main overview rows correspond to 500-request MATH500 evidence. Native DFlash-14B is absent rather than imputed; all existing native comparisons use the recorded baseline.
- Source, RelaySpec, and native throughput ratios use their matched run's AR reference. The right panel uses the separately computed source-relative paired throughput ratio, not a ratio of unrelated AR-normalized aggregates.
- The right panel's progress ratio is correctly calculated as RelaySpec recorded progress per cycle divided by source-reuse recorded progress per cycle. It is a descriptive point estimate with no manufactured confidence interval. Keep the caption's qualification that this is **not a timed cycle decomposition** or direct proof of which kernel saves time.
- All twelve budget fits use 512 records. All matched-budget decoding groups contain 16 requests; budget values correspond to the nominal warm-training times. A separate setup/cache charge must stay explicit in the caption.
- All nonfeature budget ratio intervals remain below parity; the largest upper endpoint is approximately 0.860. Thus the plotted “below parity” statement is true for these conditional request intervals, while general claims about optimized CE/LoRA should remain bounded to these tested recipes.
- Two LoRA curves represent update-initialization variants sharing the initial mapper. “Fit 1 / fit 2” is acceptable; avoid calling them fully independent training pipelines or a quantified fitting-seed uncertainty estimate.
- Block-size endpoints correctly show RelaySpec throughput approximately 3.69×, 4.70×, and 2.93× AR at blocks 8, 16, and 32, with recorded progress approximately 4.98, 6.48, and 4.14 tokens per cycle. The block screen is separate from the 500-request primary result and uses **naive** source reuse, correctly distinguished by the legend.
- Color/shape encodings are readable without relying on color alone. The horizontal main overview avoids cramped grouped bars and clearly shows missing native evidence.

## Caption requirements for integration

- Main overview: 500 requests per target/family, matched BF16 runs, paired 95% request intervals, source-relative progress is descriptive, no native DFlash-14B measurement.
- Warm budget: 16 exposed development requests, 256-token output cap, 512 training examples, setup/cache construction excluded from warm-training budget, CE/LoRA charged for initial mapper training, intervals exclude fitting-seed and selection uncertainty.
- Blocks: separate 64-request DFlash-8B development screen, proposal block lengths 8/16/32, naive source-reuse control, request-conditional intervals and recorded progress-accounting definition.

No code or figure edits were made in this review. Findings were sent to the root for corrections before final rendering.

## Follow-up PNG inspection

The root corrected the budget y-axis/evidence field to end-to-end throughput and shortened the block progress label; both corrections are visible and pass inspection. Moving the block legend to the upper-left removes the block-8 overlap but places its third row over the block-16 RelaySpec point/curve. The root has been advised to add top y-axis space (approximately an upper limit of 6.5) or place a shared legend outside the axes. Main and budget PNGs otherwise pass visual inspection.
