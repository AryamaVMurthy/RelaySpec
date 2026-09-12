# Existing manuscript evidence inventory

This records the existing manuscript and its source definitions. It is not the
new experiment plan and does not authorize additional runs. The current suite
must finish before that plan is issued, as recorded in NEXT_PAPER_SCOPE.md.

Manuscript: `paper/iclr2027/relayspec_iclr2027.tex`.
Source SHA256: `799e084e028b9752d91932854ada4303be62b37123a0e1270183378de015c6df`. Its README identifies the
September 9 manuscript as the current version. The inventory below concerns
completed/reported studies, rather than every proposed matrix found in scripts.

| Existing study | Recorded design | Manuscript/source location |
|---|---|---|
| Primary Qwen coverage | DFlash and DeepSpec EAGLE-3, Qwen3-4B source to Qwen3-8B/14B; AR, source reuse, available native controls | Manuscript lines263 and604; `configs/relayspec_protocol.yaml` |
| Workload breadth | MATH-500:500; GSM8K:128; HumanEval:164; MBPP:378; MT-Bench:80 two-turn conversations. Historical primary timing is sequential per worker, not batch128 | Manuscript line278; generated workload tables |
| Extended data scaling | Eleven counts:16,32,64,128,256,512,2048,4096,8192,16384,32768. All receive8192 batch-four updates =32768 record presentations; single fitting seed | Manuscript line383 and1490; `generated/expanded_numina_data_table.tex`; `scripts/report_record_scaling.py` |
| Earlier distinct-data/compute curves | Separate historical MATH data:64–4096 records at1024 batch-four updates. One continuous4096-record fit saves128,256,512,1024,2048 updates | Manuscript line822; `generated/controlled_data_table.tex`; `figures/controlled_fitting.pdf` |
| DFlash-8B capacity | Thirty fits: N512/2048 × dense plus factored-linear/GELU-MLP widths64,128,256,512,1024,2048,4096.8192 batch-four updates;1024 validation records; seed1729 | Manuscript line1535; `scripts/build_capacity_paper_assets.py`; `configs/submission/scaling/matrix-focused-v1/matrix.json` |
| Longer optimization | Width512 factored-linear and MLP, both N512/2048, continue to32768 updates with optimizer/RNG/data-position continuity | Manuscript line1591; `figures/numina_epochs.pdf` |
| Regularization | Forty-two nonzero arms: N512/2048 × dense/factored512/MLP512 × L2{1e-7,1e-6,1e-5,1e-4} or AdamW decay{1e-4,1e-3,1e-2}. Zero-penalty controls come from capacity fits | Manuscript line1614; `scripts/build_regularization_paper_assets.py` |
| EAGLE-3/8B replication | Fourteen primary fits: N512/2048 × dense plus factored/MLP widths128,512,2048. Historical extra dense seeds are separate | Manuscript line1672; `generated/eagle_small_capacity_table.tex` |
| 14B replication | Fourteen primary fits per drafter: N512/2048 × dense plus factored/MLP widths512,1024,4096. Historical extra dense seeds and four EAGLE learning-rate follow-ups are separate | Manuscript line1888; `generated/target14_dflash_table.tex`, `target14_eagle3_table.tex`, `rate_check_table.tex` |
| Block length | DFlash8B blocks8,16,32; historical64-request/512-cap screen. Family extension also reports coarse6–32 and finer Llama9/10/11, cross15/16/17 screens | Manuscript lines804,2088,2851 |
| Matched adaptation budgets | N512; feature regression, connector token CE and rank32 drafter LoRA. Three-rate selection, budgets0.25/1/4 ×91.413 measured optimizer seconds. CE/LoRA initializer training is charged. The paper retains two LoRA seeds | Manuscript line2012; generated timed-budget tables |
| Composition | Math-only2048 versus1024 Numina+1024 Dolly; dense, factored512, MLP512.8192 batch-four updates; common validation1024 math+1024 instruction. Historical second dense seed is separate | Manuscript line2387; generated composition tables |
| Richer generated-rollout calibration | DFlash4B→8B;16384 Numina rollouts, three layer+context calibration epochs;128 requests/cap2048; native comparison with reversed GPU assignment | Manuscript lines494 and2719; `generated/rollout_transfer_table.tex` |
| Llama and cross-tokenizer extension | Llama3.1-8B drafter→Llama3.2-3B; Qwen3-4B drafter→Llama3.1-8B. Historical nested4096/8192/16384, epochs1/3/6/12 and targeted24-epoch extensions. Final128 MATH/cap2048 uses FP32 target/maps and BF16 drafters | Manuscript line2812; `generated/family128_table.tex` |
| Other reported analyses | Layer selection, compression, native-interface studies, frozen GSM8K confirmation, input-only task complexity, output/acceptance diagnostics, numerical precision, memory and GB10 profiling | Respective appendix sections and included generated tables/figures; remaining source details still need inventory |

## Source checks already resolved

- The expanded data curve really uses8192 batch-four updates, rather than the
  older1024-update budget. Its report builder rejects a claimed record count
  unless `distinct_records_seen == distinct_examples`. The32k point therefore
  has32768 presentations and a checked full pass; do not confuse it with the
  earlier smaller-data curve.
- The pinned primary manuscript uses five feature taps for both its DFlash and
  DeepSpec EAGLE-3 interfaces. Source/8B taps are[1,9,17,25,33], target14B taps
  [1,10,19,28,37]. Use these actual releases, rather than assuming a generic
  EAGLE checkpoint's feature count.
- Some proposed generic scaling matrices contain unrun or superseded cells.
  Completed paper builders/registries determine the reported matrix. In
  particular, the completed penalty study uses N512/2048, not the older generic
  script's N512/32768 proposal.
- Public PARD/SD² measurements use their own runtimes. The manuscript explicitly
  qualifies this comparison; they are not matched-vLLM batch128 controls.
- The original paper documents numerical AR disagreement in ordinary BF16
  Transformers execution. Exact agreement for the pending standalone checks
  must be measured, not assumed from the vLLM result.

## Current AUF recipe facts relevant to interpretation

`matrix/model.py` loads the ZIP map checkpoint for five-map CE/AUF. These are
token-loss refinements of calibrated maps, rather than random-start AUF fits.
The current transfer metadata states4096 initializer records and three ZIP
calibration epochs for Qwen/Llama; cross-family uses4096 and one epoch. The
subsequent token-loss fit is one epoch in each case. Original-interface CE uses
its own original-MSE initializer. Current audit/report output now exposes this
distinction. Any later data-efficiency accounting must include initializer data
and calibration cost, not only records seen by the refinement stage.
