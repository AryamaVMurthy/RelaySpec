## Latest update — approximately22:49 IST

- Normal RelaySpec matched controls implemented and queued31579/31580/31581,
  including explicit input RMSNorm in vLLM and a runtime numerical probe.
- Six CPU tests pass, including the actual original feature loss and the
  normalization-before-projection runtime path. GPU execution still pending.
- Cross-tokenizer audit31582 completed: unequal vocabularies and69/132 probes
  with different token counts. Direct unchanged handoff port is unsupported;
  an explicit alignment/verification extension remains separate pending work.
- Qwen14 cache31529 completed; Llama31530 live. Origin31525 has completed its
  AR/native128 evaluations and is evaluating the trained mapper (last50/128).
- Normal controls currently await shared account GPU quota. Existing live GPU
  jobs have not been restarted or interrupted. Updated dependencies preserve
  the four-GPU total as documented in the plan and job manifests.

# Current state — 2026-09-11, approximately 22:40 IST

- Original handoff Math fit: **complete**, 2,000 optimizer updates, 16,000
  record presentations, 860,160 trainable parameters; 231.266 seconds on two
  GPUs. This is fit time, not data preparation or evaluation time. Its full
  128-request AR/native/mapper evaluation remains live in31525.
- Qwen8 transfer cache31528: complete,4,096 records verified/repacked.
  Qwen14 cache31529 live (last observed3,584 records); Llama31530 follows.
- Both transfer architectures implemented with unchanged packaged AUF and
  training loop, explicit source/checkpoint validation and folded exports.
  Four CPU tests pass, including finish-reason disagreement detection.
- Original pending gates31551/31552 were canceled before running to include
  deployment tests. Replacement gates31553/31554 test both Qwen8 variants.
  The first now follows origin31525 rather than waiting for slow old31346.
- Full fits queued:Q8 r56/five31556/31557; Q14 r56/five31558/31559;
  Llama r56/five31560/31561. Every full fit includes a short GPU training,
  frozen/export and four-request token/finish-exact decoding gate first.
- Matched vLLM evaluation jobs31562–31573 cover AR, ZIP, handoff-r56 and
  handoff-five for all three pairs:128 requests, max2,048 tokens, natural
  EOS,three timing repetitions. Two serial one-GPU lanes start after both
  origin31525 and old31346 finish. Training uses one serial two-GPU lane.
  Thus the campaign remains within four GPUs including old live work.
- Evaluation and collection dependency IDs are recorded under reports/.
  No new handoff transfer speedup result is available yet.

## Remaining scope (goal is not complete)

Normal RelaySpec matched control; heterogeneous-vocabulary cross-family
implementation/correctness gate; other main workloads; data, optimization
and capacity sweeps under the new recipe; selected fitting-seed confirmation;
full GPU profiling; final Transformers cross-checks; comprehensive evidence
and new paper. ZIP and normal RelaySpec remain distinct baselines.

## Earlier audit and scheduling history

# Handoff transfer status

This campaign follows the exact handoff.zip source, not the earlier AUF
five-map or rank32 drafter-LoRA recipes.

- Archive SHA25650ef7120aee5b9087356814eee4e235281437a6fa99ba5097ff5d86932002d45.
  All37 hashes and the unmodified packaged CPU loss check pass.
- Pinned SpecForge953d43a0c1c0f5e32989dc43f91ce5fc2d9ddfef checked out and
  copied to the remote control directory. Vendor source is unchanged.
- Original Qwen3-4B target/drafter revisions exactly match the handoff and
  were reused. The pinned Math target adapter was downloaded separately.
- CPU preparation31522 COMPLETED exit0. PEFT0.20.0 and accelerate1.12.0
  installed in an isolated overlay; existing shared runtime not changed.
  Torch/vLLM/Transformers match the handoff; pinned SpecForge imports.
- The author's original feature cache path is permission-denied. No access
  workaround attempted; our own fidelity trajectories/features will be generated.
- Gate31523: pending AssocGrpGRES, requests2 GPUs on node07 after31450.
  Runs32 generated responses, dense capture, exact packaged2-update DDP fit,
  frozen-weight/export checks, then4 AR/native/mapper exactness checks.
  First-four eval checks are preliminary and do not by themselves establish
  comprehensive long/EOS/length-stop coverage; add representative cases
  before declaring the full fidelity gate complete.
- Full origin31525 follows31523:4096 records,2000 updates, then128x2048
  AR/native/mapper comparisons against bundled references.31524 was replaced
  before execution to include the full-checkpoint frozen/export verification.
- No new GPU handoff result is available yet. Transfer ports, normal-RelaySpec
  matched fitting and three-way transfer evaluations still need implementation.
- Held all21 earlier pending arrays (31503,31502,31500,31498,31496,31495,
  31494,31492,31491,31490,31489,31488,31487,31486,31481,31480,31479,31478,
  31472,31471,31470). Their outputs and definitions remain intact. Existing
  live jobs were allowed to finish; old results retain their recipe labels.

Plan: docs/plans/2026-09-11-handoff-transfer.md. Shared maximum remains4 GPUs;
external account quota can delay allocation even when fewer study GPUs run.

### Five-map AUF variant added

Implemented both `fusion_r56` and `five_maps` transfer parameterizations.
Five maps restore the ZIP epoch-3 W_i checkpoint; the fusion residual uses
the same checkpoint's folded export. Frozen-body/export checks, initial
folded-weight equivalence, trainable-key checks and per-job GPU telemetry
are included. Three CPU tests passed (unchanged training contract, AUF
chunk loss/gradient equivalence, five-map folding/freezing/gradient flow).

Qwen8 GPU gates submitted:31551 fusion residual,31552 five maps, sequential
and dependent on cache preparation and completion of older evaluation31346.
These are pending gates, not completed experiments or measured speedups.
Origin gate31523 completed successfully. Origin full31525 remains running.
Q8/Q14 dense-feature repacking31528/31529 ongoing; Llama31530 waits on31529.
Full 128x2048 comparison results for the new pair are not available yet.
