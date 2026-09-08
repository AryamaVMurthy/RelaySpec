# Graphical revision: independent scientific/editorial plan

Inspected manuscript: `paper/iclr2027/relayspec_iclr2027.tex`, current nine-page main text; generated evidence registries and tables; existing plot builders. Review date: 8 September 2026.

## Main-text diagnosis

The current main text has two figures (workflow and data scaling) and seven numerical tables. It describes several visual findings—saturation, capacity/fitting mismatch, the acceptance/cost tradeoff, adaptation-budget dominance and asymmetric transfer gains—without showing their shapes. These are the priority graphics. Decorative charts or a figure for every small control would weaken the nine-page narrative.

The four numbered contributions must remain in the introduction. Each should lead naturally to one main figure: interface workflow; small-data/capacity frontier; frozen-drafter native-throughput comparison; family transfer with asymmetric gains. A primary speed/source-removal overview establishes practical value before the finer findings.

## Recommended main-text allocation

| Location | Graphic | Required content | Replacement/space control |
|---|---|---|---|
| Method, approximately p. 3 | Existing workflow schematic | Frozen source and target during fitting; only map updated; source transformer absent during generation; target verifier remains authoritative | Retain existing compact figure. |
| Primary results, approximately p. 4–5 | Two-panel performance/mechanism overview | (a) Four primary MATH-500 RelaySpec AR-speedups with paired intervals and native/source controls where measured; (b) source→relay mean progress and throughput relationship, or a clearly isolated historical source-cost profile | Replace main throughput table; retain full exact table in appendix. Do not borrow AR from an unrelated run. |
| Small-data/capacity results, approximately p. 6 | Three-panel calibration figure | (a) 16–32,768 records at fixed 8,192 updates; (b) parameter count versus decoded throughput for matched linear/MLP maps; (c) frozen 512/2,048-example confirmation for both families with 95% retention threshold | Replace existing single-panel scaling figure and frozen-confirmation table. Put all exact values/counts in appendix tables. |
| Matched adaptation, approximately p. 7 | Warm-budget line chart | All four methods at 22.9, 91.4 and 365.6 seconds, including both LoRA seeds; clearly labeled 16-question, 256-token development screen and excluded setup/cache costs | Replace the three-column adaptation table. Could share a row with the calibration figure only if labels remain readable. |
| Extensions, approximately p. 8–9 | Native/family multipanel | (a) Both native-relative Numina runs and paired intervals, same fitted map, AR reused; (b) original versus selected Llama and Qwen→Llama operating points, with paired relative-change intervals | Replace rollout and family tables. Retain separate panel headings for batch-invariant BF16 and FP32 target/BF16 drafter. Never share an absolute-TPS ranking axis across these runtimes. |

This yields five substantive main figures plus the workflow if budget/extension are separate, versus two previously. Preserve the small primary quality table or a compact zero-centered paired accuracy-difference plot. The heterogeneous public-baseline table remains valuable for explicit runtimes/correctness; converting it to one ranked TPS bar chart would obscure rather than improve positioning.

## Appendix visual coverage priorities

1. **Task complexity:** difficulty, input-length and optionally subject small multiples using the existing `task-complexity-results.json` registry. Show all method cells or a principled subset with the complete table alongside; label group sizes (difficulty n=42/21/65; input length n=52/57/15/4). Use point/interval plots without lines implying continuous causal relationships. A four-request subgroup must be visually marked as exploratory. Denominator is dense N=2,048 within each group, not AR. Difficulty, answer length and capping are associated; this is not a causal complexity experiment.
2. **Quality versus numerical agreement:** a zero-centered paired accuracy-difference forest plot for the four primary MATH-500 cells, paired with exact-token agreement fractions only if separately titled. Accuracy intervals include zero; low BF16 token agreement cannot be hidden by the extension's 128/128 FP32 agreement. Head-only precision diagnostics are a separate small experiment and do not demonstrate FP32-target equivalence.
3. **Selective-capture memory scaling:** memory saved versus input tokens (2,273, 8,993, 16,826, 31,526), two/five taps, from the generated capture table/raw evidence. Show full-state and selected-state peak allocation or their difference. Mark one constructed prompt per length, two order reversals; this is allocation scaling, not long-context quality. Keep source-transformer removal memory on RTX 6000 Ada separate from selective-capture memory on L40S.
4. **Block-size response:** plot AR, native and relay throughput for blocks 8/16/32 in the matched 64-question historical screen. Mark trained block 16; no invented interpolation/tuned best beyond measured settings. Family block tuning should be a separate plot because its prompts/output cap/runtime differ.
5. **Native interface compression:** connect trainable parameter count or number of feature taps with retained native throughput in separate retargeted, native-DFlash and native-EAGLE panels. Show frozen GSM8K confirmation plus code's third-layer benefit. Separate initialization-with-calibration from training-free slicing; inherited columns are an initialization, not zero-cost training. Include the negative smaller-verifier boundary.
6. **Training composition:** retain the existing throughput plot and add feature-validation versus decoded-throughput evidence only if raw points expose the mismatch clearly. Plot two seeds as separate markers rather than hiding them behind a mean; separate math/code/chat by panel. Mixed composition has a math-speed tradeoff and should not be sold as uniformly better.
7. **Optimization and regularization:** existing `numina_epochs`, `numina_regularization`, `eagle_small_epochs`, `target14_dflash_fitting` and `target14_eagle3_fitting` already cover trajectories. Improve style/legends if necessary, but do not duplicate them. Parameter-count curves must distinguish actual measured decoded points from validation-only cells.
8. **Acceptance/source-cost accounting:** keep position-wise source/relay survival curves; if adding source-cost decomposition, use its historical run's profiler rows, explicit hardware and measured-region labels. End-to-end throughput cannot be algebraically reconstructed from progress alone; `c/a` is a decode approximation and the profiler ideal is not measured speed.

## Scientific and visual acceptance criteria

- Each figure is generated reproducibly from recorded numerical evidence and emits source paths/hashes; no values copied from narrative text without a checked source.
- Each panel identifies target/drafter, evaluation workload/count, output cap where relevant, throughput definition, comparison denominator and uncertainty unit.
- Main text uses end-to-end throughput including prompt processing; decode-only data are explicitly labeled and not silently mixed.
- Within-family curves use a stable palette and marker style: RelaySpec/dense prominent, alternatives distinguishable without color; parity/95% reference lines labeled once.
- Precision/runtime changes, original-versus-selected recipe changes, train-data count versus optimization work, and inherited training versus marginal fitting cost remain explicit.
- Do not connect nominal categories with a smooth trend. Record/parameter axes may use log scale with explicit powers-of-two ticks. Bar axes start at zero; zoomed ratio point plots may center on parity.
- Both positive and negative completed findings remain visible: cross-family gain and Llama null; code source-reuse boundaries; low BF16 exact agreement; warm-budget control losses.
- Uncertainty is paired request bootstrap unless specified otherwise; repeated execution/GPU reversal does not become an independent fitting seed or independent prompt sample.
- At final manuscript scale, axis labels/legends should be approximately 8–9 pt or larger; no overlaps, cut-off annotations, stretched figures or orphan captions.
- The final PDF must still have exactly nine main-text pages, with the contributions block and limitations retained, all labels/references resolved, all new figures included and actual rendered pages inspected.

## Review status

Planning/evidence audit complete. New figures and final render have not yet been reviewed; this document is not a visual signoff.
