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
- Fourteen unit tests: exact AUF first-failure supervision, detached support,
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
  This early gate has been refined by the fitted-interface diagnostic in NUMERICS.md;
  FP32 argmax invariance remains required, while BF16 support differences are reported.
- Decode31252 stopped at an obsolete telemetry attribute before measurements.
  vLLM0.28 V2 uses `speculator`/`get_draft_model`; hook updated and rerun31254
  succeeded, including original source embedding/head attachment checks.

## Completed main fitting and current evidence

Q8 dense target capture31258 and separate1024-record validation31261 are
complete. Main4096-record,3-epoch seed42 fits are complete: ZIP31263 (lr1e-3),
AUF31267 and CE31268 (lr1e-4). These are initial configurations; equal-budget LR
selection is still required. Offline evaluations31271/31272/31273 use4096 fixed
blocks from1024 separate validation records:

| Objective | Epoch3 CE | Epoch3 AUF | Mean accepted draft prefix |
|---|---:|---:|---:|
| ZIP | 1.3584 | 0.4312 | 5.8091 |
| AUF | 3.4561 | 1.5490 | 1.6875 |
| CE | 3.0037 | 1.6184 | 1.5679 |

These are offline teacher-context measurements, not decoding TPS. Initial AUF
has not improved on ZIP. New128-request,2048-cap speed results remain unrun.

Llama preflight31270, paired pilot data31274, three32-record pilot fits31275,
and vLLM diagnostic31279 completed. All4 outputs match AR, but tiny mapped fits
are slower: AR46.22TPS, AUF40.30, CE42.16, ZIP41.90. Preserve the untied source
head; incorrect inherited source BOS/EOS metadata is corrected in exports.
Q14 pinned download31297, preflight31315, pilot data31318 and pilot fits31321
completed. Streaming family fitting passed all six family/loss GPU cells31323, including
folded-export checks.

The benchmark overlap audit31289 found no exact matches, and four potential
MATH template overlaps in the full16384-record calibration pool:31,58,226,336.
Exclude these from new benchmark subsets; this is not semantic decontamination.

## Active and queued work

- Matched rank56 fusion-LoRA AUF/CE screen31320 starts from the same frozen
  ZIP4096 endpoint and trains on512 existing records for3epochs. Charge its
 4096-record ZIP initialization. Only A/B train; original targets remain frozen.
  Initial31319 stopped at BF16 batch-shape argmax sensitivity; see NUMERICS.md.
  Both completed31320. Epoch3 offline prefix: AUF5.8318, CE5.8154 versus F0
  ZIP5.8091. The small AUF difference is about0.39%, not a demonstrated speedup.
  Revised gate measured exact FP32 argmax and zero BF16 AUF-support differences
  on this diagnostic, despite one BF16 greedy difference out of30 positions.
- LR array31316: ZIP/CE/AUF x1e-4/3e-4/1e-3,512records/3epochs,1024 validation,
  at most2 GPUs. It is now running. Selected4096 refits31328 follow it,
  then two-shard128x2048 Numina development decoding (ZIP/CE/AUF/native/AR).
- Llama4096-record paired capture31290: eight512-record shards, one GPU.
  Family validation31325 follows it, using separate1024 archived validation
  problems rather than training or final benchmark prompts.
- Q14 main data31324 follows successful streaming trainer gate31323; eight
 512-record shards, one GPU. Validation31327 follows Q14 data. Llama main fits31333 follow validation31325;
  Q14 main fits31334 follow validation31327. Both also require the completed
  Q8 LR screen archive31335, and transfer its fixed LR choices across families.
- Node07 uses at most2 of our GPUs; node06 uses at most2. Dependencies enforce
  the four-GPU campaign cap. Check live Slurm state for current execution.

## Remaining

Finish equal-budget learning-rate comparison, ZIP-initialized continuation,
128×2048 decoding, additional families and compatibility gates, scaling and
ablations, required repetitions/confirmation, hardware profiling, final two
Transformers backend comparisons, and the separate evidence-backed manuscript.
Full task remains active. Check live Slurm state rather than treating this file
as a live job monitor.

Our reported Qwen3 experiments use a shared tokenizer and token-ID vocabulary;
the mapper adapts hidden representations or the fusion interface, not vocabulary
IDs. Consequently, these results do not demonstrate heterogeneous-vocabulary
DFlash support.
