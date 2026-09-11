## Verified fit and scaling preparation update

Priority scheduling update:Q8 AUF evaluation tasks31606_2/3 materialized as
31648/31649 and their unrelated-family dependencies cleared after both fits,
CE,and normal Q8 finished. Q14 two-GPU fit31559 now waits for these two tasks;
normal Q14 may use1GPU alongside old Q14's1GPU and Q8 evals'2GPUs,total<=4.
Remaining pool tasks keep original dependencies. Slurm rejected an initial
circular update; corrected by clearing the two task dependencies first,then
adding31559's dependency on their materialized IDs. No running work canceled.
CE31586 completed0:2,000 updates,538.65 seconds,export verifier passed.

Implemented separate large-data fixed-epoch recipe:4K/8K/16K/32K,three-epoch
matched ZIP initialization then four complete AUF epochs,both architectures,
per-epoch checkpoints. A1,024-update sweep would expose at most8,192 record
presentations and cannot claim full16K/32K training. This large-data study
therefore reports variable compute explicitly. Packing supports verified
prefixes of canonical manifests; trainer count handling reused unchanged.
Only the first4K fit has been submitted; remaining large fits/evals to wire.

32K data-only chain enabled:31488 generates only added16,384 records after
31486;31489 captures frozen source/target features after31488;new CPU2
assembly31645 follows31489. Each GPU array max2 tasks,serial data stages.
Generation batches up to64 sequences,cap4096; evaluation remains128/cap2048.
Original16K rollouts/paired features reused and reserved dev/eval groups
excluded by prepared manifest. Old-loss training arrays remain held. Larger
data fits/evaluations still pending implementation and data verification.

Capacity collector31644 queued after31643. It retains every declared rank,
all timing repetitions,trainable parameter counts and fitting costs; compares
against matched AR and primary rank56,checks common runtime/request manifests,
and fails on token/finish disagreement. Capacity remains pending runtime work.

Capacity evaluation31635 follows rank8 fit31634,with128/cap2048/3repetitions
and full token/finish checks. Ranks16/32/128/256 fits/evals31636–31643 form
a serial chain after successful rank8 evaluation. Rank56 reuses the primary
measurement. Explicit handoff-capacity runtime label avoids calling these
rank56 results. At most2GPUs in this lane,plus possible2GPU capture. Final
capacity aggregation remains to wire; none of these ablations has run yet.

Q8 five-map31585 completed0; fit644.74 seconds,2,000 updates,export verified.
Proof archived in reports/q8-handoff-five-fit. Uniform CE31586 now running.
Separate rank-capacity port supports8/16/32/56/128/256 while preserving the
main rank56 implementation; test verifies rank56 arithmetic unchanged except
variant naming. Rank8 fit31634 queued after both profiling arrays31632/31633,
2GPUs,4krecords/2kupdates. This is a capacity ablation,not exact rank56 recipe;
its decoding evaluation and other rank fits remain to schedule.13 CPU tests pass.

Scaling preparation now hashes each reused full shard once,retaining that
verified digest for its symlink instead of rereading the same tensor file.
Main comparison collector now fails its job on any token or finish mismatch
while preserving the diagnostic report. Added a regression case with identical
tokens but differing finish reasons; all12 CPU tests pass.

GPU profiling arrays31632(Q8,node07) and31633(Llama,node06),eachmax1GPU,
queued after31629. AR/normal/ZIP/BA/five-map,4 requests capped2048; separate
instrumented outputs have timing_valid=false. Traces limited to64 engine
iterations plus1sGPU telemetry; neither full-run SM occupancy nor valid TPS.
Combined profile lanes plus possible2GPU capture remain within4GPUs.
Q14 normal31580 dependency is cleared but Slurm reportsAssocGrpGRES; do not
restart or bypass account limits. Main active jobs continue advancing.

Full small-sweep CPU collector31630 scheduled after31629. It requires all
nineN16–4096 points,128 exact outputs/finishes across3 timing repetitions,
verified1,024-update/8,192-presentation AUF fits,and retains training costs.
The sweep fixes AUF work only:three-epoch ZIP initialization cost grows withN
and must be charged separately; do not describe total optimization as fixed.
Normal Q8 job31579 completed0. Next normal Q14 job31580 is scheduler-pending
withAssocGrpGRES; other live work continues,so no restart is needed.

Normal Q8 all three timing repetitions archived:172.8651,172.9689,173.0015TPS;
mean172.9451, sample timing SD0.0713. Full outputs and finish reasons identical
across repetitions; matched AR verification still pending. See timing-summary
in reports/q8-normal-partial (directory name retained from initial archive).

Remaining small-data fits31614–31621 and evaluation arrays31622–31629 submitted
forN32/64/128/256/512/1024/2048/4096,matched per-N ZIP initialization,both AUF
architectures,1,024 updates. They wait for successful N16 end-to-end collection
and both compute arrays. Fits run serially on2GPUs,then evaluation arrays run
serially with max2GPUs each. Dense capture may use2 more; total stays<=4.
These are queued work,not completed evidence; result collection remains to wire.

Compute evaluations31612 (Q8,node07,max1GPU) and31613 (Q14,node06,max1GPU)
queued after31609. Each evaluates both architectures at500/1000/1500 updates
with128 requests,cap2048,three repetitions; final2000 endpoint comes from
primary evaluations. Checkpoint/export verifier now accepts explicit paths
for saved intermediate endpoints. These are continuous-fit checkpoints under
the same2000-update schedule,not independently retuned short schedules. No
new training or AR generation. Combined with capture's2GPU limit,total<=4.

Archived normal Q8 repetitions0/1 in reports/q8-normal-partial. Each contains
128 complete timing-valid rows and122,768 output tokens; aggregate rates
172.8651 and172.9689TPS (mean172.9170). Third repetition and matched AR
agreement are pending. These are provisional absolute normal-baseline rates,
not a measured AUF improvement. Q8 five-map fit last420/2,000 updates.

Dense-feature packer now accepts explicit record counts/data directories and
initializer exports while retaining4096 defaults. Added a32-record packing
test that also rejects a mismatched capture-manifest hash; eleven CPU tests
pass. Larger-data packing remains pending dense captures and matched ZIP fits.
Q8 five-map31585 passed its gate and entered full2,000-update training.

Loss-independent dense-feature capture array31486 released with replacement
dependency afterok:31606:31607 (24 tasks,512 records/task,max2 GPUs).
It completes records4096–16383 from existing frozen-target rollouts; the first
4096 captures are already verified and reused. This is data capture only,
not release of any old-loss training array. Concurrent small-data fit/eval
uses at most two more GPUs,so the combined maximum remains four. Larger-data
packing/fits/evaluations still require follow-up after capture verification.

Q8 fusion job31584 completed0,2,000 updates in633.58 seconds; export verifier
passed (onlyA/B train,frozen non-fc exact,folded relativeMSE0.0). Proof archived
in reports/q8-handoff-r56-fit. Q8 five-map31585 has started. Full transfer
throughput remains pending. Scaling collector now checks all three timing
repetitions,runtime/manifest identity and128 token/finish comparisons; ten
CPU tests pass,including deliberate runtime mismatch rejection.

Scaling fit31608 queued on node07,two GPUs,afterok both evaluation pools31606
and31607. First point:N16,three-epoch ZIP initialization fitted on exactlyN16,
then1,024 AUF updates for each fusion-r56 and five-map architecture. Immutable
job-specific source/output paths; verification required per export. This job
does not include decoding evaluation. Broader scaling waits for this first
runtime check; no old-loss scaling arrays were released.
Evaluation array31609 now follows31608:three methods (matchedN16 ZIP, BA,
five maps),128 requests/cap2048,three timing repetitions,up to two GPUs.
Reuses existing matched Q8 AR references and requires full token/finish
agreement for every repetition. No new redundant AR generation is scheduled.

Q14 fusion AUF job31558 completed 2,000 updates in707.98 seconds on two GPUs.
Archived summary and verification in reports/q14-handoff-r56-fit: only A/B
train, frozen non-fc weights exact, folded export relative MSE0.0. This is
training/export evidence; its full128-request throughput evaluation is pending.
Q8 retry31584 passed the repaired unique-path gate and reached1,310/2,000
updates at the last live check;31579 and31346 remain active evaluations.

Added prepare_scaling.py to build isolated matched-data inputs. It requires
an exactly N-record, three-epoch ZIP initializer, checks checkpoint and capture
hashes and exact sequence identity, reuses full mmap shards and materializes
only partial shards (including N16). A separate count-aware trainer preserves
the original main4096 recipe. This prepares inputs only; the scaling GPU runs
are not yet launched. Nine local CPU tests pass, including rejection of an
initializer fitted on more records than the declared scaling point.

## Evaluation scheduling and timing update

Pending evaluation jobs31562–31573,31587/31588 replaced (none were running)
by pools31606 (Q8,node07,two GPUs maximum) and31607 (Q14/Llama,node06,two
GPUs maximum). They wait for the current training,normal-control and old
jobs,then use four GPUs across independent method evaluations. Each GPU
still measures sequential requests;128 requests,cap2,048,three repetitions
are unchanged. Collection dependencies repaired to the new arrays.

Planning estimate from measured rates:main queued comparison matrix about
6–10 more hours with balanced pools;full remaining scaling,confirmation,
profiling,backend checks and paper provisionally24–48 hours. These are not
finish guarantees; queue waits,validation failures and extension engineering
remain uncertain. Measured fits:origin3.85 minutes,normalQ8 4.72 minutes,
Q14 AUF11.8 minutes. Long AR references dominate remaining GPU time.

## Current execution update — approximately23:10 IST

Four GPUs active:Q14 AUF fusion full training31558 (last880/2000 updates),
Q8 normal evaluation31579 (first repeat96/128),older Q14 evaluation31346
(AR81/128). Q8 normal fitting completed in283.36 seconds; runtime input
normalization matches the numerical reference exactly in the GPU probe.

Both Q8 short AUF architecture gates passed4/4 token/finish checks. Subsequent
full jobs31556/31557 failed before full training: repeated setup overwrote
diagnostic exports and strict checkpoint-hash checks correctly rejected reuse.
The repaired script uses job-specific diagnostic paths; full retries31584/31585
are queued. Q14's first gate passed and its full fit is running.

Matched Q8 uniform-CE control31586 uses the same ZIP initialization,rank56,
2,000 updates and handoff sampler; only prefix support is removed. NativeQ8
and CE evaluations31587/31588 added. Q14 AR reuse31589 requires identical
configuration,requests,prompt IDs and complete timing-valid outputs; retains
original rows/provenance as one repetition, with two fresh repetitions still
required. Eight CPU tests pass. Detailed job dependencies are recorded in
reports/repair-and-control-jobs.json and other job manifests.

## Verified result — original handoff reproduction

Origin31525 completed successfully. All128 outputs and finish reasons match
both AR and the immutable package references. Max output2,048; all stopped
naturally,19,070 total output tokens. AR31.171 TPS; native122.752 TPS;
AUF fusion154.287 TPS,1.25690x native and4.94968x AR. Accepted draft tokens
per verification improve3.4771 to4.6982. One fitting seed and timing run;
request-bootstrap95% native-relative interval[1.2260,1.2891] excludes seed
and timing repetition variation. This is the pinned frozen-target-LoRA Math
setting, not a transfer-to-original-target result. Source logs, export check,
summary and figure are archived under reports/origin-full* and figures/.

Q8 fusion gate31553 completed with4/4 token/finish matches at128-token
cap; that diagnostic is not the main throughput comparison. Five-map gate
31554 is now running. Normal Q8 control31579 is training; first epoch
completed in96.7 seconds. No full transfer comparison is complete yet.

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
