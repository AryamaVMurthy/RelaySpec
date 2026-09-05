# RelaySpec claim–evidence map

Status values are **established** or **partially established**. The unified
two-family harness and corrected two-turn chat runs are complete.

| Claim | Status | Scope and conditions | Evidence chain | Final artifact |
|---|---|---|---|---|
| Canonical speculative verification preserves the target distribution in exact arithmetic | partially established | greedy implementation is exercised; stochastic correction must be conformance-tested before being claimed; BF16 bitwise equality is reported separately | formal induction / Leviathan et al. ICML 2023 -> target-only commit assertions -> token agreement and first-divergence diagnostics | method theorem + finite-precision exactness table |
| The relay removes the source conditioning trunk | established for DFlash 4B->8B/14B and EAGLE-3 4B->8B/14B | optimized source reuse executes only committed tokens through layers 0--33 | provider intervention -> component timers -> full-run source shares of 38.8%/30.2% for DFlash and 33.1%/27.1% for EAGLE | component breakdown |
| DFlash RelaySpec improves over optimized DFlash source reuse | established on full and preregistered-complement MATH-500 at both scales | greedy Qwen3 non-thinking, frozen 4B proposer, 8B/14B verifier | full raw rows -> 1.4831x [1.4745, 1.4917] and 1.2458x [1.2382, 1.2533], 1,000/1,000 exact source/relay outputs | `reports/final/DFLASH_8B_MAIN_RESULT.md` and `reports/final/DFLASH_14B_MAIN_RESULT.md` |
| EAGLE-3 source reuse contains a removable bottleneck | established for 4B->8B/14B | optimized committed-prefix source trunk through layer 33 | pinned DeepSpec wrapper -> full paired component profiles -> 26.8--33.7% cross-task source shares and 20--31% isolated peak-memory savings | `reports/final/BREADTH_MATRIX.md` and `reports/final/EAGLE3_*_MEMORY.md` |
| The same interface-transplantation mechanism works for DeepSpec EAGLE-3 chain drafting | established at both target scales | Qwen3-4B EAGLE-3 proposer relayed to 8B and 14B | pinned wrapper -> conformance -> matched architecture ablation -> confirmatory 1.250x [1.244, 1.256] at 8B and 1.081x [1.076, 1.087] at 14B | second-family main table |
| Task quality is unchanged relative to matched source reuse | established on MATH, GSM8K, HumanEval, and MBPP for both families and scales | full-target verification; pinned Qwen math and EvalPlus scoring | DFlash has exact paired breadth outputs; EAGLE finite-precision differences retain identical paired official math/code quality; every mismatch is logged | `reports/FINAL_RESULTS.md`, EvalPlus summaries, and mismatch audits |
| Coefficient-free relative interface MSE is a sufficient core objective | established as the selected objective for both proposer families and both target scales; global optimality is not claimed | EAGLE and DFlash relay fitting | norm/cosine identity -> loss unit test -> matched DFlash comparison: 1.523x/1.265x versus historical 0.1 loss at 1.509x/1.251x -> exact verified live runs | `reports/design-selection/dflash/OBJECTIVE_SELECTION.md` and architecture ablation |
| Gains are governed by removable runtime share and retained acceptance | established by held-out EAGLE forecasts and 16 cross-task same-run diagnostics | source share, relay share, and position-wise survival are measured for every cell | development forecasts reproduce EAGLE scaling; cross-task model gets 16/16 point-estimate directions with 0.34% mean absolute relative error | `reports/final/BREADTH_MATRIX.md` and `reports/amdahl-analysis.md` |
| A coefficient-free provider rule avoids promoting measured slowdowns | established descriptively across the completed matrix | workload-level calibration; no per-request oracle claim | require Amdahl prediction >1 and paired speed lower bound >1 -> source fallback on both 14B code cells and inconclusive EAGLE-14B chat -> 1.1354x equal-cell geometric mean | profile-policy section of `reports/final/BREADTH_MATRIX.md` |
| Relay adaptation uses much less target-generated data than target-specific proposer training | established for reported data volume; compute-cost superiority remains scoped | 4,096-example fits, 86.7/118.5-second measured wall time; no cross-hardware extrapolation | state-dict count -> measured fit telemetry -> DFlash ~800K examples and EAGLE-3 ~532K entries from primary papers -> 195x/130x data ratios | `reports/adapter-cost.md` |
| The result generalizes across proposer mechanisms and target scales | established for the reported batch-one scope | DFlash parallel diffusion and pinned DeepSpec EAGLE length-7 autoregressive chain; Qwen3-8B/14B; math/code/chat | all four full MATH cells pass positive paired speed with unchanged official accuracy; 16 breadth cells expose and correctly predict the task/scale boundary | headline table and `reports/final/BREADTH_MATRIX.md` |

## Source-status constraints

- DFlash is accepted at ICML 2026. The checked evidence is its camera-ready
  arXiv record plus the official ICML 2026 program; no proceedings-volume
  number is claimed before publication.
- EAGLE-3 is NeurIPS 2025, peer reviewed.
- DFlare, SpecForge, and other 2026 arXiv work are labeled preprint/software as
  appropriate.
- HyperDFlash (arXiv:2606.26744v2, June 2026) is the closest newly checked
  target-interface-alignment neighbor. It trains a new DeepSeek-V4-specific
  DFlash using the target's inherited hyper-connection reducer; it neither
  keeps an old proposer frozen nor removes a separately executed source trunk.
  It narrows, but does not duplicate, the interface-transplantation claim.
- AngelSpec (arXiv:2607.25852, July 2026) changes and co-specializes proposer
  architectures and allocates verification effort online. It reinforces the
  need for task-specific cost/acceptance reporting, but does not transplant an
  unchanged proposer across targets or eliminate its old source trunk.
- RepSpec (ICLR 2026) changes target-specific drafter training through
  inference-time-merged structural re-parameterization. It is a peer-reviewed
  training-efficiency/acceptance neighbor, not frozen-proposer transplantation.
- SPEED-Bench (ICML 2026) establishes that task mix, context, concurrency, and
  serving engine can change speculative-decoding conclusions. RelaySpec's
  current four-rank statistic is therefore labeled batch-one micro-rate; no
  production-serving throughput claim is made.
- Speculative KV Coding is a technical note and is used only in the novelty
  audit.
- Quantitative RelaySpec claims must point to immutable raw artifacts, not to a
  project summary alone.
