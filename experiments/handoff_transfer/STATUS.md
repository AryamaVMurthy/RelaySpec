## 2026-09-12: 512-anchor request implemented as isolated pilots

Latest requested scope: Qwen8, Llama, and Qwen-to-Llama. Same-family scripts
use 512 anchors, 4,096 cached records, rollout cap4,096 and planned2,000
updates/globalbatch8 (=16,000 presentations). LM-head loss chunk16 preserves
AUF summed numerator/denominator; CPU loss and gradient equivalence tested.

- 31727: Qwen8 two-update, fold and4-request/cap128 exactness gate, node07.
- 31728: Llama equivalent gate, node06, after31727.
- 31734: cross labels attempt failed before generation because Llama8 assets
  are node06-local.31735 resubmitted on node06; this is rollout and tokenizer
  alignment work only, not cross training or a throughput result.
- Main pools31606/31607 also wait31728 and31735. Max four GPUs preserved.
- Full training/eval scripts prepared, not submitted before memory/timing gates.
- New cross alignment utility tests exact shared text prefixes, rejects lossy
  Unicode and uses source labels with target context strictly before anchor.
  Cross training/inference still needs implementation and validation.
- Existing Llama eight-anchor BA completed2,000 steps in555.195 fitting seconds.
  Existing Q14 oldAUF finished128/cap2048 at67.0738TPS,128exactAR outputs;
  this is the older recipe, not the new handoff512-anchor method.

See ANCHORS512_PLAN.md. New512-anchor arms have no benchmark results yet.

## 2026-09-12 01:00 IST: live status audit and additional Q14 evidence

User requested end-to-end status. Livejobs31560 LlamaBA2GPU (~340/2000updates at00:59),31580normalQ14second timingpass1GPU,31346olderQ14AUFevaluation1GPU (68/128requests when checked). Allnode06,fourGPUs. Llama gate exactness JSON confirms4/4token+finish equality; gate uses legacy path gate-fusion_r56 in originally submitted script, so an attempted archive using the newer job-suffixed path failed without affecting running work.

Discovered completed olderQ14AR/ZIP/CE128-cap2048 passes and archived them in q14-older-completed. Verified runtime and manifest match current normalQ14, plus128/128fulltoken/finish equality to AR for every method. AR16.57624TPS; normal111.52430(6.72796xAR); ZIP111.31218(6.71516xAR); olderCE73.53547(4.43620xAR). One seed/one timingpass; olderCE is not matched handoffCE. OldAUF still partial and excluded. NewQ14BA/five full decoding remains pending.

## 2026-09-12: final Transformers result collection queued

CPUcollectors31721(Q8 after31696)/31722(Llama after31697) now summarize both gate4/cap128 and full128/cap2048 standalone Transformers outputs. Require correct family/mode/repetition, backend/runtime consistency, identical primary checkpoint hashes/manifests, and full token/finish equality to the Transformers AR baseline. Report one timing repetition and do not compare raw TPS between backends. Synthetic test validates successful collection and rejects checkpoint mixups. Scripts syntax checked; actual Transformers runs remain pending at the end of the queue. LlamaBA is still in its decoding gate without reported failure at last check.

## 2026-09-12: complete Q14 fitting cost comparison

Archived Q14 ZIP fitting summary and generalized measured cost ledger. Normal1115.416s=18.59min, ZIP1137.869s=18.96min, ZIP+AUFBA1845.847s=30.76min/0.7094GPUh, ZIP+AUFfive1909.900s=31.83min/0.7450GPUh. BA continuation trains1,576,960params but preceding ZIP trains65,536,000; ledger explicitly counts both phases and excludes rollout/capture/packing/loading/gates/queue/evaluation. Existing Q8 rows and evidence hashes reproduced exactly after generalization. LlamaBA two-update training gate completed1.4235s; export/decoding gate still ongoing when checked.

## 2026-09-12: Q14 five-map full training complete and verified

Job31559 completed all2000updates/16000record presentations in772.031363seconds (12.8672minutes) on2GPUs. Export verifier passed: exactly the five fc.maps weights trained, folded relativeMSE0, every non-fc exported tensor unchanged. Full summary/verification archived in reports/q14-handoff-five-fit. This completes both Q14 AUF fits (BA already complete); full decoding comparison still pending. Success released LlamaBA job31560, now running on2GPUs alongside normalQ14 and olderQ14evaluation (four total). No job restarted.

## 2026-09-12: completed timing-repeat figure generated

Generated and visually checked figures/q8_three_repeats.png/pdf directly from the verified three-pass comparison. Shows absolute TPS by repetition and all three paired percentage changes, explicitly labeled one fitting seed/development only and AR/ZIP pending. Preserves positiveBA(+2.68%) and negativefive(-4.91%) outcomes. Plot source saved for regeneration; no manuscript results rewritten prematurely. Source-copy audit found only1.5MB remote reports, so no unsupported claim that copying causes slow training; current Q14five reached1580/2000updates.

## 2026-09-12: Q14 first complete normal evaluation archived

Q14normal r0 completed128requests/cap2048:111.524295TPS,122902generated tokens,1102.019963summed requestseconds. Archived full rows/summary and SHA256 metrics in reports/q14-normal-first-pass. One fitting seed/one timing repetition so far; no AR or AUF comparison claim. Q14five reached1390/2000updates.

Final-stage command audit found checkpoint digest used whole-file reads, risking memory exhaustion for large Llama exports in4GB CPUallocations. Changed confirmation freeze/runner shared digest to streamed8MiB chunks, preserving SHA256 results. Focused freeze/runner tests pass and remote source synced. Queued job scripts correctly reference modules through control PYTHONPATH.

## 2026-09-12: final confirmation and Transformers stages queued

Freeze CPUjobs31687/31688/31689 wait for main comparisons, workloads, small/large scaling, capacity, compute, profiles, three-seed reports and matched drafter control. Confirmation arrays Q8=31690,Q14=31692,Llama=31694 run serially by family, each0-11%2 across four workloads/three repetitions. CPUcollectors31691/31693/31695 follow. Transformers replications Q8=31696,Llama=31697 wait for all confirmation collectors and therefore run last, oneGPU each. No new jobs running now; current4GPUlimit retained. Complete submission record persisted incrementally in reports/final-jobs.json; queue_final refuses duplicate/partial resubmission.20handoff tests pass, wrappers syntax checked. Final-stage runtime validation and all outcomes remain pending; manuscript rewrite/repro package still required after results.

## 2026-09-12: final confirmation Slurm wrappers prepared

Added freeze_confirmation.sbatch (CPU2 per family), confirmation.sbatch (oneGPU task; workload/repetition array), and collect_confirmation.py. Confirmation uses frozen checkpoints/engine settings, isolated source snapshots, rawGPU telemetry, per-repeat protocol hashes and full-output checks. Collector requires all four reserved workloads and three repetitions with runtime equality to the frozen protocol; reports clearly distinguish confirmation from development and seed variation. Shell and Python syntax checks pass. These final-stage jobs are not yet submitted; they must follow all required development/seed/scaling/profiling results, then Transformers runs last. Q14five reached900/2000updates at latest check.

## 2026-09-12: concurrent benchmark JSON race fixed

Found shared pilot_data.write used a fixed .tmp filename; concurrent workload repetitions preparing the same manifest/provenance could replace each other's temporary file and fail. Changed to unique per-write temporary names with atomic rename and cleanup. Concurrent test exercises64writes/eightthreads and validates complete JSON plus no temporary leftovers. Confirmation materialization also uses separate data-rN paths.21tests pass including20handoff checks. Synced writer for future jobs; currently running jobs retain their snapshots. Q14five reached720/2000updates; normalQ14 full evaluation still underway. No extra GPU work launched.

## 2026-09-12: reserved confirmation runner prepared

Added confirmation_runner.py. It requires a frozen protocol and exact checkpoint/config hashes before materializing reserved prompts, preserves all prespecified primary arms, rotates mode order by timing repetition, launches each vLLM engine in a separate subprocess for clean memory teardown, verifies runtime config equals the frozen development config, and retains full AR token/finish diagnostics for all128requests/cap2048. Output links pin protocol hash per repetition and prevent mixing frozen protocols. Validation test rejects changed weights before evaluation.20tests pass overall and focused freeze/runner validation passes. No GPU confirmation run or Slurm submission yet; scheduling remains after required development evidence. Q14five reached520/2000updates at last observation.

## 2026-09-12: confirmation freeze checks validated

Added per-family freeze support so Q8 checkpoints can be checked on node07 and Q14/Llama on node06 without assuming a shared scratch namespace. Seed evidence now requires exactly42/43/44 and three repetitions per seed; empty or incomplete maps cannot pass. New tests exercise immutable output, changed-checkpoint rejection and absence of a frozen output on failed validation.20tests pass. Confirmation runner and combining per-family protocols remain pending; no confirmation data touched. Q14five training reached330/2000updates at latest live check.

## 2026-09-12: confirmation freeze preparation

Added freeze_confirmation.py to pin all prespecified primary seed42 arms after complete three-repeat main/development-workload reports and Q8/Llama three-seed evidence. Requires exactness, family/protocol consistency and checkpoint hashes matching development measurements; writes exclusively and refuses overwrite. Preserves all normal/ZIP/BA/five arms (plusQ8native), avoiding selection on confirmation. Explicitly does not claim confirmation of every exploratory scaling/rank point. No confirmation prompts evaluated and no confirmation job submitted; runner integration and final sequencing remain pending. Q14five31559 reached120/2000updates after its decoding gate; four GPUs still allocated.

## 2026-09-12: Q14 normal fit archived; future I/O timing added

NormalQ14 completed4096-record/three-epoch fit:1115.416283seconds (18.5903minutes),1893updates over1,292,174 sampled positions,65,536,000parameters; export relativeMSE0. Training summary/contract and hashed cost evidence archived in reports/q14-normal-fit. First full evaluation reached64/128 when checked. Q14five31559 passed through its AR gate and remains active; no restart.

Added feature-batch wait timing to future train_normal invocations without changing batch order, RNG, objective or optimizer. Logs separate time obtaining each CPU batch from the remaining fit loop; explicitly not GPU-only kernel time. Existing active jobs use their original source snapshots. This instrumentation addresses the current lack of evidence attributing the longer family fit time to I/O versus computation. Syntax validation passed; no timing claim made from unrun instrumentation.

## 2026-09-12: Q8 three timing passes complete; Llama seeds queued

Q8 normal mean172.945143TPS; BA177.579371TPS, paired mean ratio1.02679599 (+2.6796%), timingstdev0.04077TPS; five164.455648TPS, ratio0.95091237 (-4.9088%),stdev0.03581TPS. All three repetitions128/128 full output/finish equality versus normal,122768tokens per pass. Archived all rows/summaries/hashes in q8-handoff-three-repeats. AR/ZIP/native/CE are still pending. Q14five job31559 started on two GPUs after Q8 tasks completed.

Llama seed43 fit/eval31680/31681 and seed44 fit/eval31682/31683 queued on node06 after Q8seed4431678. CPUcollector31684 follows. Small-data31608 now waits for31683, preserving experimental lane maximum2GPU plus data lane2GPU. Same fresh ZIP/normal/BA/five setup and full128/cap2048 three timing repeats; Llama target paths/tokenizer and cached records are retained. Shared seed collector now audits both schemas and full provenance;19tests pass. None of the added Llama seed jobs has run yet.

## 2026-09-12: Llama initializer schema audited for seed replication

Audited completed Llama ZIP summary: contract schema, objectiveZIP, frozen targets, 4096records/3epochs, ordered part manifests and checkpoint hash; status explicitly says fit complete/offline validation pending. Added shared initializer_contract reader to prepare_scaling/prepare_seed; Q8 schema remains separately checked. Family manifests are loaded in numeric part order and every SHA256 verified before existing exact dense-row identity checks and checkpoint verification. Seed preparation now supports Llama43/44, but launch/evaluation/collection wiring remains pending. Updated synthetic Q8 fixture to include the real required family field;19 tests pass. Q8 BA third repetition finished; five-map still live, so full three-repeat joint collection waits.

## 2026-09-12: across-seed aggregation queued

CPU2 collector31679 follows seed44 evaluation31678. It requires original ZIP, normal, BA and five-map fits with matching seed42/43/44 provenance, records/epochs/updates, verified frozen weights, and exact evaluated checkpoint hashes. Uses shared matched AR references and full128/cap2048 three-repeat equality checks. Averages timing repeats within each fit before computing sample standard deviation/min/max across three seeds; does not count nine timing runs as independent fits. Per-seed reports now receive correct seed metadata rather than hardcoded42.18 tests pass. Actual seed runs and Llama seed support remain pending.

## 2026-09-12: Q8 full-pipeline seed jobs queued

Seed43 fit/eval31675/31676, seed44 fit/eval31677/31678; serial chain starts after workload31661. Small-data fit31608 now waits for31678, preserving at-most-two experimental GPUs alongside the at-most-two-GPU data lane. Each seed independently fits ZIP and normal for three epochs on4096 cached records, then BA/five for2000updates/global8/eight anchors. AUF two-step verification gate precedes full fitting. Evaluations rotate all four methods across repetitions,128/cap2048 with full token/finish equality against matched main AR. No new target inference for data is needed.17 tests pass and shell scripts parse. GPU seed runs, Llama seed wiring and across-seed collection remain pending.

## 2026-09-12: full-pipeline seed provenance preparation

Added prepare_seed.py for Q8 seeds43/44. It requires a fresh4096-record/three-epoch ZIP initializer whose recorded seed matches, verifies exact cached sequence identity through existing prepare_scaling, and generates an isolated AUF trainer. Seed variation covers ZIP/AUF initialization, anchor sampling and shard shuffle. Main seed42 trainer remains unchanged. The generated trainer rejects an inconsistent --seed and fixes previously hardcoded seed42 summary wording for these extra runs. Normal baseline must also refit with the matching seed. No seed jobs launched yet; Llama initializer schema and launch/evaluation wiring remain pending. All17 tests pass. Live Q8 third repetitions progressing; no duplicate launches.

## 2026-09-12: hardware profiling collection wired

CPU-only collectors31673(Q8 after31632) and31674(Llama after31633) require all five methods, four completed instrumented requests/cap2048, profiler max64 iterations, nonempty GPU kernel traces and nvidia-smi telemetry. Both compressed and plain PyTorch trace formats are supported. Reports retain kernel duration categories, busy-time union, top kernels and source hashes; scope explicitly excludes primary throughput, full-request profiling, SM occupancy and unverified attribution of whole-node GPU telemetry. Test rejects profile outputs labeled as valid primary timing. All16 tests pass. Active four GPU jobs still progressing; normalQ14 was loading its next engine at last check.

## 2026-09-12: second complete Q8 timing repetition archived

Both AUF methods now have two complete 128-request/cap2048 repetitions archived in reports/q8-handoff-two-repeats. BA:177.533103/177.594983 TPS, mean177.564043, paired mean ratio to normal1.02687468 (+2.6875%). Five maps:164.495466/164.445378 TPS, mean164.470422, ratio0.95115276 (-4.8847%). Matched normal172.865057/172.968865 TPS. All methods emitted122768 tokens per pass and match normal128/128 full token sequences and finishes. Collector verifies summary completeness, matching runtime/manifest/family/mode/repetition and timing validity; archives include SHA256 provenance. Third AUF repetitions still running when checked; AR/ZIP/native/CE still pending. These are one-fit development results, not confirmation or a 10% gain.

## 2026-09-12: matched drafter-LoRA campaign queued

31669 (two-GPU fit) waits for main evaluation pools31606/31607; 31670 (one-GPU evaluation) follows; CPU2 collector31671 follows evaluation. Workload array31659 now waits for31670, keeping the experimental lane at most two GPUs alongside the at-most-two-GPU data lane. Rank32 body LoRA uses the same frozen ZIP interface, 4096 records, 2000 updates/global8, eight anchors, chunk2, lr1e-4 and seed42. Two-step training/merge gate precedes full fit; four-request cap128 AR gate precedes full128/cap2048 three-repeat evaluation. New handoff-draft mode prevents conflating it with historical draft-auf. Collector compares normal/ZIP/BA/five/body with matching runtime/manifests and requires full AR equality. All15 tests pass; scripts syntax-checked; runtime validation remains pending.

## 2026-09-12: matched drafter-body control prepared, not yet GPU-validated

Audited historical drafter LoRA: 4096 records, three epochs/384 updates, logical batch32, four anchors, selected lr2e-5, rank32/9,175,040 parameters. It is not a matched handoff control (2000 updates/global8/eight anchors/lr1e-4). Added separate draft_control model/train/merge verifier: same ZIP F0 frozen; AUF objective and training loop unchanged; LoRA placed on seven attention/MLP projections per draft block. Rank32 repeats the historical capacity; optional rank4 is the nearest lower parameter count to fusion rank56 (not exact parameter matching). All body updates merge into dense weights; verifier requires fc/head/embedding/norm and every other non-adapted export tensor unchanged. Source tests pass; actual GPU forward, gradient, merge and 128/cap2048 evaluation remain unrun. No additional GPU jobs submitted.

## 2026-09-12: compute-scaling aggregation wired

Added CPU-only collectors 31667 (Q8, after 31612) and 31668 (Q14, after 31613). They require all 500/1000/1500/2000-update checkpoints for both architectures, three timing repetitions, 128 requests at cap2048, matching runtime/manifest and full AR token/finish equality. Checkpoint training provenance must show the same 2000-update schedule; the endpoint reuses main measurements. Reports retain cumulative AUF time and explicitly exclude shared ZIP initialization cost. This is continuous-fit checkpoint analysis, not independent budget optimization. Synthetic validation covers finish mismatch and schedule rejection. Four existing GPU jobs remained running; no extra GPU allocation submitted.

## Verified fit and scaling preparation update

First-pass paired request analysis:BA faster91/128,5k paired bootstrap ratio
interval[1.0202,1.0342];five-map faster30/128,interval[0.9431,0.9603]. These
exclude seed/repeated-timing uncertainty. Mean accepted draft tokens per
verification normal6.1919,BA6.3885,five5.8353; measured total request wall
per verification about41.5ms for all (includes prefill,not kernel latency).
This is consistent with acceptance driving the speed difference,not proof
of a causal mechanism. Source hashes,report and viewed figure archived.

FIRST COMPLETE Q8 AUF128/cap2048 PASS (repeat0):BA177.5331TPS,+2.7004%
versus normal172.8651TPS;five-map164.4955TPS,-4.8417%. Both128/128 full
token and finish matches to normal,122,768 output tokens. Common runtime and
manifest checked. This is one of three repetitions,not finalAR/ZIP/native/CE
comparison. Archived in reports/q8-handoff-first-pass. Do not claim10% gain.

Workload collectors31664/31665/31666 scheduled on matching family nodes.
Main collector now validates family,method,and repetition identifiers as well
as runtime/manifest consistency; workload collector reuses that validation.
Q14 normal31580 has started; four GPUs active at the last live check.

Measured Q8 fitting-cost ledger and figure generated from archived summaries.
Normal4.72min,ZIP4.82min,ZIP+AUFBA15.38min,ZIP+AUFfive15.57min,
ZIP+uniformCE13.80min sequential fitting. GPU-minutes charge two-GPU
continuation separately. Excludes rollout/capture/packing/loading/gates/queue;
not total adaptation cost. Figure viewed and legend overlap corrected.
See reports/q8-fitting-costs.json and figures/q8_fitting_costs.{png,pdf}.

Frozen workload development arrays31659(Q8),31660(Q14),31661(Llama) queued
serially after both main pools. Each4workloads x3timing repeats,128/cap2048,
AR/normal/ZIP/BA/five plus Q8native; all full token/finish checks mandatory.
Small-data gate31608 now waits for31661 so workload breadth precedes scaling;
data-preparation chain can use2 GPUs alongside workloads'2,total<=4.
Confirmation split remains reserved. No old-loss workload arrays released.

Prepared standalone Transformers runner/final script for Q8 and Llama. Uses
pinned official DFlash plus existing verified relay generator,folded exports,
normal-input RMS only for normal arm,and its own synchronized AR reference.
Short4/cap128 gate must pass for all methods before128/cap2048 runs. One
timing repetition per secondary backend replication; primary vLLM uses3.
These files pass syntax checks but have NOT run on GPU; scheduling remains
last after remaining vLLM/workload/confirmation work,as user requested.

Large-data CPU collector31657 queued after31656. Requires all fourN points,
four complete epochs per architecture (including expected microbatch counts),
verified exports,all128 token/finish matches across three timing repetitions,
and includes one-GPU ZIP initialization plus two-GPU AUF fitting costs.
Q8 main AUF evaluations remain live; BA first repetition passed16 requests
at the last check. No partial request log is treated as a full benchmark.

Large four-epoch fits now fully queued:4K31647,8K31650,16K31651,32K31652;
full evaluations31653–31656 follow with128 requests/cap2048/3repetitions.
32K fit additionally waits for assembly31645. Updates areN/2 and record
presentations4N per architecture,with shared matched three-epoch ZIP init.
Final large-data collection remains to wire. CE fit proof archived in
reports/q8-ce-fit. Q8 BA/five main evaluations are producing responses and
acceptance counters; no complete128-request AUF result has been promoted yet.

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

## 2026-09-12 02:10–02:16 IST matrix continuation

Four Qwen100-update/512-anchor fits verified. Dense-fusion CE took340.96s;
five-dense CE342.03s. Both train52,428,800 parameters,800 record presentations,
rank0actual175,645 anchors/400records. These are fitting costs, not decoding gains.
Cross16-record CE/AUF integration fits completed31776; export/frozen-weight checks
passed. Full cross-data study and decoding correctness are still outstanding.

Repaired missing node-local reference/manifests and Transformers BatchEncoding
serialization in validation. Screens31784/31785 replace failed31778/31779.
Queued additional LR fits31780/31781 and all-candidate validation31782/31783.
Four-GPU maximum is enforced through two2-GPU fitting lanes or two2-wide1-GPU
validation arrays; old baseline pools31606/31607 remain explicitly held until
new full-fit dependencies are installed. Routine next poll not before02:30:47.

## 2026-09-12 02:31 IST verified screen and cross gate repair

Qwen fiveBA56 AUF,100updates,512anchor limit:149.0766TPS versus24.5635AR,
6.0690×,32/32 full-token and finish matches,cap512,15,940 actual output tokens.
This is a tuning screen, not the final128/cap2048 comparison and not evidence
of improvement over normal RelaySpec. CE screen31785 was running at check.
Llama normal job31581 completed. Cross gate31777 failed during synthetic
vLLM non-greedy sampler warmup; explicit warmup-only handling added and31800
resubmitted. Real requests remain greedy-only and target verified. Node06 has
2.1TB free at check; node07 has11TB. Next routine check not before02:51:02.

## 2026-09-12 02:51 IST

Four GPUs actively fitting initial cells:31771_0 onnode07 and31772_0 onnode06,
two each. Cross gate31800 passed both CE/AUF4/4 token+finish checks,cap128.
Pilot TPS:AR24.69,CE29.20,AUF30.33; these are two-update integration results.
Qwen fiveBA CE100-update screen:146.03TPS,5.953×AR,32/32 exact,cap512,
versus AUF149.08TPS/6.069×AR. No normal-RelaySpec gain established yet.

Cross target and references staged onnode07. Paired-feature pilot31803 follows
Qwen initial fits. Cross full pipeline31806–31819 queued after same-family finals,
with at most4 GPUs onnode07 and strict stage checks. Additional old baseline
pools31606/31607 remain held to prevent overlap. Broad grid fitting is substantial:
Qwen39600 optimizer updates at observed~3.4sec/update implies~37hours on its
two-GPU lane before evaluation; this is an extrapolation, not a full-study ETA.
Next routine poll not before03:11:10IST.

## 2026-09-12 03:11 IST

Four GPUs running:31771_2 node07,31772_3 node06,two each. Initial100 fits
continue. Llama normal-initializer CE/AUF finished245.08/245.65 trainingseconds
respectively,800 presentations,512anchorlimit. Qwen normal-initializerAUF fit
finished340.49seconds and exportverification passed; enclosing31771_1 shell
then failed on NFS stalefilehandle. Retained these completed artifacts.
Llama31772_2 failed with same shellreaderror after exact4-request decodinggate,
before100fit. Recovery31842 queued directly as a Slurm-spooled script;
31781now waits for its success. FourGPUbudget preserved. Avoid replacing
shared shell scripts while active wrappers are reading them.

Separatebatch4/8measurement and comparisonprotocol implemented/deployed;
61CPUtests pass. GPUbatchvalidation remains outstanding. Final128/cap2048
benchmarks remain queued after tuning; no new finalspeedclaim. Next routine
GPUcheck not before03:31:11IST.
