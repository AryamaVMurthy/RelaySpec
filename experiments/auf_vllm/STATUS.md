# AUF study implementation status

Updated 2026-09-11 during implementation. This is a separate study in branch
`research/auf-vllm-20260911`; the original paper and results remain unchanged.

## Resolved protocol

Use the original frozen paper targets for teacher labels, feature capture,
verification, and AR. No target LoRA is trained or loaded. Main AUF trains the
same five maps as ZIP, using only current-prediction accept-until-fail CE.
Fusion LoRA is an optional later ablation. See the full plan in
`docs/plans/2026-09-11-auf-vllm-study.md`.

Main reported decoding: 128 distinct requests/workload, maximum 2048 output
tokens, natural EOS. Main seed42 checkpoints receive three timing repetitions;
extra fit seeds43/44 are limited to the Q8 and L3 robustness cells. No new
paper-level performance claim has been established yet.

## Completed

- Turing preflight31246: L40S, CUDA GEMM, torch2.13.0+cu129,
  vLLM0.28.0, transformers5.16.1, pinned Qwen assets and rollout inventory.
- Twelve unit tests: exact AUF first-failure supervision, detached support,
  microbatch normalization, ZIP objective/folding, LoRA initialization,
  no future features, EOS/padding, and attention positions.
- Pilot data31248: 32 archived Q8 training rollouts, eight separate development
  rollouts from frozen Q8, dense Q8 and Q4 features. Wall time4m38s.
- Pilot fits31251: AUF, uniform CE, ZIP; one epoch,32 records, seed42,
  lr1e-4, one record/anchor per optimizer step. Frozen-drafter gradients and
  BF16 folded exports verified. These settings are diagnostics, not the main fit.
- vLLM diagnostic31254: four requests,128-token cap, all five methods matched
  AR on4/4 complete outputs. AR24.60TPS, native128.47, AUF25.60,
  CE25.64, ZIP24.68. No meaningful AUF advantage is resolved here.
- Batch profile31256: four anchors per record, longest pilot records,
  records/microbatch1/2/4/8 all passed. Throughput11.3–11.5 records/s;
  peak allocation at four records16.86GB, at eight31.44GB. This profiles
  forward/backward only, not data loading or optimizer/checkpoint overhead.
  Retain4 records ×4 anchors,8 accumulation steps for main token training.

## Numerical/runtime issues resolved

- Fit31249 failed the initial BF16 batch-vs-individual1% relative-L2 gate.
  Diagnostic31250 found BF16 relative-L2 about1.34%, FP32 about1.76e-6,
  and exact greedy agreement. Padding perturbation and official-helper
  comparison were exact. The documented gate now requires FP32 relative
  MSE<1e-10, exact supervised greedy agreement, and BF16 relative-L2<2%.
  This is numerical tolerance, not a claim of bitwise BF16 batch invariance.
- Decode31252 stopped at an obsolete telemetry attribute before measurements.
  vLLM0.28 V2 uses `speculator`/`get_draft_model`; hook updated and rerun31254
  succeeded, including original source embedding/head attachment checks.

## Active preparation and queued main work

- Q8 target dense capture:31258 resumes verified files from31253. Four records
  per capture call; manifest hash computed once. Batched full-prefix vs truncated
  prefix checks must pass relative MSE<1e-6. Original31253 was intentionally
  stopped to remove per-record manifest hashing and batch the calls.
- Offline validation31261: reserve1024 records from the archived ZIP evaluation
  pool indices128:1152, excluding original train/dev and the legacy first128
  evaluation questions. These are explicitly validation, not fresh confirmation.
  Generate frozen-Q8 labels in128-record shards with64 active sequences, then
  capture dense target features. Fixed-epoch fitting may overlap preparation;
  validation is still required before configuration selection.
- ZIP main reference31263:4096 matching records,3epochs,lr1e-3, original
  quarter-position cache and equal-example weighting. No new source capture
  is needed. Redundant source capture31255 was stopped and queued31259 cancelled.
- AUF31267 and CE31268:4096records,3epochs,lr1e-4, seed42,4×4×8 batch contract;
  depend on target capture31258. These are initial main configurations, not
  learning-rate-selected final checkpoints. Replaced dependency-only31264/31265
  to overlap fixed fitting with validation-label preparation.
- Llama assets/runtime staging to node06 is in progress. Storage check31266
  confirmed node07 scratch is not accessible directly from node06; explicit
  node-local staging is required. No Llama AUF run has completed yet.

## Remaining

Finish main Q8 fits, offline validation, equal-budget learning-rate comparison,
128×2048 decoding, additional families and compatibility gates, scaling and
ablations, required repetitions/confirmation, hardware profiling, final two
Transformers backend comparisons, and the separate evidence-backed manuscript.
Full task remains active. Check live Slurm state rather than treating this file
as a live job monitor.

Our reported Qwen3 experiments use a shared tokenizer and token-ID vocabulary;
the mapper adapts hidden representations or the fusion interface, not vocabulary
IDs. Consequently, these results do not demonstrate heterogeneous-vocabulary
DFlash support.
