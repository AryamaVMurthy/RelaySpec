# RelaySpec iterative research — 7 September 2026

Active goal: discover promising applications, mechanisms and native speculative-decoding ideas, then strengthen the paper with verified findings. No campaign stopping threshold has been reached. The previous goal turn made progress: four bounded experiments completed, follow-ups launched, and completed data-scaling evidence integrated.

## Resources and protocol

Use four L40S GPUs on Turing node07. Each GPU runs an independent screen under an external540-second timeout plus10-second kill grace. All four lanes are short, satisfying the requirement that at least two finish within ten minutes. Every intervention has a paired unmodified mapper and native/AR controls. Screens use eight previously evaluated GSM8K questions and128-token output caps. The frozen-confirmation outputs remain historical evidence; these reused questions are now explicitly development data for new ideas. New confirmations must use separately checked, unexposed questions and frozen selection.

No task-quality claim follows from short-output exact matches. Memory is co-resident campaign memory; fake quantization is BF16 perturbation, not an integer implementation. SVD is randomized with recorded seed/rank/power iterations, not an exact full spectrum.

## Wave1 (28391), completed

| Lane | Hypothesis | Runtime | Result / decision |
|---|---|---:|---|
| 0 | Post-fit compression retains decoding better than fitting a narrow map from scratch. | 120.1s | Passed artifact checks; detailed results below |
| 1 | A small subset of target layers carries most of the transferable predictive signal. | 115.7s | Passed artifact checks; detailed results below |
| 2 | Tiny-data and small-data maps lie in a connected useful alignment region. | 117.1s | Passed artifact checks; detailed results below |
| 3 | Post-fit compressibility transfers from diffusion to autoregressive drafting. | 173.9s | Passed artifact checks; detailed results below |

### Lane0

- native_ar: 38.14 tokens/s, 29.1% of paired base, 1.000 mean progress/cycle.
- native_target_dflash: 141.13 tokens/s, 107.6% of paired base, 5.557 mean progress/cycle.
- relay_base: 131.14 tokens/s, 100.0% of paired base, 4.879 mean progress/cycle.
- relay_svd1024: 123.20 tokens/s, 93.9% of paired base, 4.528 mean progress/cycle.
- relay_svd256: 36.00 tokens/s, 27.5% of paired base, 1.269 mean progress/cycle.
- relay_svd512: 62.70 tokens/s, 47.8% of paired base, 2.236 mean progress/cycle.

### Lane1

- native_ar: 37.95 tokens/s, 29.0% of paired base, 1.000 mean progress/cycle.
- native_target_dflash: 140.97 tokens/s, 107.6% of paired base, 5.557 mean progress/cycle.
- relay_base: 131.02 tokens/s, 100.0% of paired base, 4.879 mean progress/cycle.
- relay_drop1: 124.15 tokens/s, 94.8% of paired base, 4.619 mean progress/cycle.
- relay_drop17: 131.52 tokens/s, 100.4% of paired base, 4.892 mean progress/cycle.
- relay_drop25: 97.82 tokens/s, 74.7% of paired base, 3.550 mean progress/cycle.
- relay_drop33: 93.52 tokens/s, 71.4% of paired base, 3.366 mean progress/cycle.
- relay_drop9: 131.07 tokens/s, 100.0% of paired base, 4.874 mean progress/cycle.

### Lane2

- native_ar: 38.33 tokens/s, 29.1% of paired base, 1.000 mean progress/cycle.
- native_target_dflash: 141.17 tokens/s, 107.3% of paired base, 5.557 mean progress/cycle.
- relay_base: 131.60 tokens/s, 100.0% of paired base, 4.879 mean progress/cycle.
- relay_blend25: 90.93 tokens/s, 69.1% of paired base, 3.288 mean progress/cycle.
- relay_blend50: 114.36 tokens/s, 86.9% of paired base, 4.166 mean progress/cycle.
- relay_blend75: 127.84 tokens/s, 97.1% of paired base, 4.767 mean progress/cycle.
- relay_small: 70.17 tokens/s, 53.3% of paired base, 2.507 mean progress/cycle.

### Lane3

- native_ar: 36.79 tokens/s, 41.9% of paired base, 1.000 mean progress/cycle.
- native_target_eagle3: 91.10 tokens/s, 103.9% of paired base, 5.350 mean progress/cycle.
- relay_base: 87.72 tokens/s, 100.0% of paired base, 4.506 mean progress/cycle.
- relay_svd1024: 45.21 tokens/s, 51.5% of paired base, 2.316 mean progress/cycle.
- relay_svd256: 24.03 tokens/s, 27.4% of paired base, 1.198 mean progress/cycle.
- relay_svd512: 30.36 tokens/s, 34.6% of paired base, 1.512 mean progress/cycle.

## Decisions and next experiments

- Promote layer redundancy to joint interventions and eventually retraining with reduced tap sets. Removing layer9 or17 alone preserves DFlash progress; removing late layers25/33 hurts. Joint removal is necessary to distinguish redundancy from individually dispensable but jointly required features.
- Plain SVD256/512 is weak in both families; SVD1024 is near useful only for DFlash. Do not blindly sweep more ranks. Compare input-aware low-rank fitting/distillation and structured layer reduction next.
- Weight interpolation improves smoothly with contribution from the512-record map. This does not establish a novel decoding method; retain as a mechanism diagnostic, lower priority than layer structure.
- Wave2 job28392: DFlash joint-drop, EAGLE joint-drop, BF16 fake-quantization robustness, native-DFlash FC repacking and compression. All four lanes remain capped at540s. Native-FC repacking must be checked against the released native baseline before interpreting its compressed variants.
- Next implementation directions: fit last-two-layer maps on the cached inputs; match initial and retrained feature errors to acceptance; explore native drafter adapter-only feature conditioning if the native control passes. Seek quality preservation, applicability and causal insight as well as speed.
- Paper updated with the completed16–32768 Numina curve in the main text, full table in appendix, historical MATH curve separate. PDF compiles to38pages; full visual/manuscript audit remains pending. No exploratory wave1 result has been added as a confirmed main-paper claim.

## Prior-work leads to verify before novelty claims

- TriSpec https://arxiv.org/html/2601.23180 : existing audit identifies adapter-only frozen-drafter prior art. Correct related-work discussion before final rewrite.
- DFlare https://arxiv.org/abs/2606.02091 : layer-wise target-feature fusion; relevant to the layer-information hypothesis.
- ReTrace https://arxiv.org/abs/2608.29748 : rejected-trajectory conditioning; new direction to read before a rejected-token proposal.
- DeLS-Spec https://arxiv.org/abs/2607.07409 : independently trained local head on a frozen DFlash backbone; relevant to cheap native enhancements.

Keep every result, including negative findings. Do not label an implementation tweak novel before the relevant primary papers are read.

## Wave2 and wave3 update

Wave2 completed three screens in120–159s; native preparation failed in11s because target_layer_ids belongs to the draft object, not draft.config. Fixed from pinned upstream source; no negative scientific claim follows from that error. Wave3 passed all four lanes in111–170s.

Wave3 DFlash last-two-tap retraining:98.7% of paired base throughput and4.767 versus4.879 progress/cycle on the eight-question screen. Mapper parameters drop from52.43M to20.97M (60% reduction). Last-one-tap reaches82.5%, so one layer is materially weaker. EAGLE last-two reaches93.5%; baseline progress differs slightly across campaigns, so an explicit duplicate-map control is now scheduled before stronger interpretation.

Native FC repacking produced identical capped output hashes to the released native control on all eight questions; acceptance trajectories differ slightly, so comparison is empirical and not claimed bit-identical execution. Native rank1024 retains96.0% of repacked-control throughput; its projection has25.17M rather than83.89M parameters. This is projection compression, not a70% reduction in the whole drafter or model.

Wave4 job28400 broadens to32 exposed questions per lane: DFlash last2 vs full and trained factor1024 on GSM8K; same on MATH; EAGLE last2 with an identical-checkpoint control; native compression1024/1536/2048 with released and repacked controls. All lanes remain capped at540s. New checkpoint artifacts use node07 scratch; prior source paths remain intact.

Next decision: promote structured last-two-tap mapping only if it holds across the expanded screens; verify repeatability before interpreting EAGLE. If native compression remains promising, compare isolated projection latency and end-to-end quality, distinguish inherited native-FC packaging overhead from compression itself. Reserve fresh confirmation questions only after selecting fixed checkpoints and thresholds.

Paper now acknowledges TriSpec adapter-only prior art explicitly in the main related work (arXiv2601.23180, Section3.2 verified on2026-09-07). New citation metadata is recorded in references.bib. Main data-curve page7 rendered and inspected; full visual/audit refresh remains outstanding. Latest PDF compiles to38pages with no final-pass unresolved references.

Persistent collector: user service relayspec-autoresearch-collector-20260907.service,24h lifetime, no job submissions or automatic retries. It rereads jobs.json and writes per-lane collected summaries. Research decisions remain agent-driven under the active user goal.
