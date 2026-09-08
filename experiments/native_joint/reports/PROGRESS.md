# Goal and progress ledger

Goal: investigate the standalone native experiment in depth and seek at least10% end-to-end speedup over original DFlash, with the user now explicitly prioritizing radical speculative-decoding ideas and very fast iterations. Near parity does not meet the target. The staged requirements are in the experiment README. The goal remains active; no pilot closes it.

## Completed

- Created a separate project in `experiments/native_joint`, without importing RelaySpec code or changing its manuscript.
- Pinned native Qwen3-8B target/drafter and original verification routine.
- Added native compact student construction, offline target feature capture from raw Numina solutions, correctly shifted masked-token KL training, paired before/after decoding and source/checkpoint provenance.
- Four local tests pass: tokenizer compatibility, no future conditioning features, teacher-detached/student-active gradients, and independent student construction.
- Job **29152**: engineering failure before training because this Transformers version returned a BatchEncoding from `apply_chat_template`. Preserved logs, fixed canonical token extraction, added regression test. No scientific result.
- Job **29153**: four-lane smoke passed, eight training and two validation records, four updates, two development requests at a 64-token cap. Joint arms update both interface and draft; interface-only leaves draft weights unchanged; target has no gradients. Peak allocated GPU memory 25–38 GB. These are pipeline checks.
- Job **29154**: four-arm pilot passed, 128 training and 16 validation records, 128 updates with accumulation 2 (256 sampled blocks), eight GSM8K development requests at a 512-token cap. Joint two-tap throughput 94.9% of original; three-layer/two-tap 47.3%; additional training of original architecture 101.2%. All student capped outputs matched native in 8/8 requests. Small point gains are not yet confirmed.

## Protocol correction and recovered evidence

Job 29155/lane2 failed a checkpoint reproduction gate after training. Whole-model `.bfloat16()` had rounded both nonpersistent RoPE frequency buffers; they are omitted from state_dict and are reconstructed in FP32 on reload. A focused test reproduced this, and parameter-only dtype conversion fixed it. Original post-training results from 29153/29154/29155 are excluded from promotion (see protocol-issue.json). Training weights remain valid.

Jobs **29157** and **29158** re-evaluated the eight pilot/rate checkpoints with original buffers and paired rounded-buffer diagnostic arms. Every corrected checkpoint passed reload and duplicate gates. Corrected results are in CORRECTED_RESULTS.md; the higher target-KL learning rates reduced validation loss but worsened decoding. The original-architecture fine-tune's 1.3% point gain has an interval spanning parity.

Job **29159** passed all four lanes with the corrected training pipeline: 512 records, 32 validation records, 512 updates, two examples per update. Joint two-tap five-layer throughput retained 94.3% with target KL, 97.4% with native-drafter KL, 93.6% with equal teacher blending, and 93.0% with higher-rate blending. All 8/8 outputs matched native per arm.

Job **29160** passed all four lanes: native-KL joint five-layer at rate5e-5 retained96.2%; four-layer drop3 and drop2 retained76.8% and76.9%, improving from66.7% and72.3% before training. Matched native-KL interface-only reached97.8%. All8/8 outputs matched native. These remain eight-request, single-seed development screens.

## Latest completed evidence and active work

Job **29161** completed feature preparation: four lanes passed in approximately three minutes. Merged `data/prepared-8192.json` verifies 8,192 training + 256 validation records with unique normalized questions and cross-split disjointness. Tensor caches remain on node07 scratch; exact token IDs and hashes are preserved in the collected job artifacts.

Job **29162** completed all four lanes and all checkpoint/frozen-control gates:

| Arm | Distinct records | Updates | Student TPS | Original TPS | Ratio [paired 95% request interval] | Elapsed |
|---|---:|---:|---:|---:|---|---:|
| Joint two-tap, five-layer | 2,048 | 1,024 | 184.7 | 183.9 | 1.005 [0.958, 1.056] | 390 s |
| Interface only | 2,048 | 1,024 | 174.0 | 178.5 | 0.975 [0.932, 1.010] | 270 s |
| Joint four-layer, drop3, lr5e-5 | 512 | 2,048 | 147.9 | 184.8 | 0.801 [0.745, 0.862] | 564 s |
| Joint four-layer, drop2, lr1e-5 | 512 | 2,048 | 149.5 | 182.3 | 0.820 [0.781, 0.867] | 565 s |

All eight capped development outputs matched native per arm. The joint fit has reached near parity but does not establish a gain. Four-layer recovery improved with longer training and remains clearly slower.

Job **29163** stopped before training at a batching precision gate. Padding perturbation was exactly invariant and relative BF16 error was 0.85%, but an absolute 0.25 threshold failed (max difference 1.0 with output magnitudes around 78). Preserved all failed logs. Replaced the scale-blind absolute bound with an output-scale-relative bound and added a much stricter independent FP32 forward check; this does not change training or model computation.

Job **29164** passed all four batching smoke lanes (batch2/4/8 five-layer, batch8 four-layer) in 27–30 seconds of Python runtime. Real padded vs separate FP32 forwards differed by 0.8–0.9 parts per million in relative L2; BF16 differed by 0.78–0.85%; perturbing padded features was exactly invariant. All train/update/reload gates passed. Peak CUDA allocation was 33.4–36.8 GB. Seven local core tests pass, including loss and gradient equivalence across batched and individual objectives. These are engineering checks, not throughput evidence.

**29165 completed all four training lanes**, immutable project `/home/aryama.murthy/native-joint-20260908-v11`, config `seeds-and-8192.json`:

- lane0/lane1: repeat the 2,048-record joint native-KL fit with seeds2718 and31415; 1,024 updates, accumulation2, rate1e-5, 540-second bounds.
- lane2/lane3: matched 8,192 vs2,048 records, batch8, 1,024 updates, accumulation1, rate1e-5; 900-second bounds. Both consume8,192 blocks; lane2 must consume all8,192 distinct records.
- All four save validation-best weights separately while evaluating final checkpoints in the standard before/after comparison.

**29166 completed all four evaluation lanes**, immutable project `/home/aryama.murthy/native-joint-20260908-v12`, config `breadth-2048.json`, runner `evaluate_native.py`: evaluate the valid29162/lane0 checkpoint on8 requests each from GSM8K/MATH/HumanEval/MTBench, 512-token cap, two timing repeats, original block16 vs student block16 vs original block1 AR. Each lane is bounded540 seconds. This also serves as the first real-GPU check of the general evaluation runner. Inspect every result before treating it as evidence.

Job29165 results: the two additional2,048-record seeds retained99.14% and98.85% throughput, versus100.46% for seed1729 in29162. All8/8 outputs matched native per seed. The matched batch8 fits retained97.85% at8,192 records and97.67% at2,048; both consumed8,192 blocks, and the large fit consumed all8,192 distinct records. Thus this matched larger-data screen shows no clear gain. All validation-best/final/checkpoint artifacts are collected.

Job29166's repeated eight-request breadth results are in EVALUATIONS.md. Student/native throughput: GSM8K100.4%, MATH95.9%, HumanEval90.2%, MTBench97.1%. All student outputs matched native in32/32 requests across both timing repeats. Native vs block1 AR identity was4/8,3/8,1/8,1/8 respectively. This is a baseline verifier/precision distinction, not evidence of quality preservation against AR. The exact cause of those baseline differences should be probed on matched divergent prefixes before a general guarantee is claimed.

**29181 completed all four lanes**, remote immutable v13, `taps-and-position-loss.json`: joint three-tap native-KL; two-tap native-KL gamma2; two-tap native-KL uniform; two-tap hard target labels. All2,048 records,1,024 updates, accumulation2, rate1e-5, firsttwo bounded540seconds (allfour use540seconds).

**29182 was canceled while pending (zero runtime), following the user’s shift to rapid radical decoding experiments**, remote immutable v14, `native-controls-and-ce.json`: original five-tap/five-layer architecture target-KL additional training across seeds1729/2718/31415, plus two-tap data-CE. Same2,048-record/1,024-update work budget and540-second bounds. These native controls are essential to separate compression benefits from additional training.

The29181 final screens retained100.2% (three taps),98.7% (early gamma2),96.0% (uniform weights), and91.0% (hard target labels). All runs completed in406–429seconds. No training arm reached the user’s10% requirement. Additional native training controls remain unrun and are deprioritized while inference-only hypotheses use unchanged native weights.

**29184 completed20 inference-only hypotheses on four GPUs**, immutable v15, `radical-screen.json`, runner `run_radical.py`. Each hypothesis used two requests per GSM8K/MATH/HumanEval/MTBench (eight total), cap256; individual tests finished in roughly25–32seconds. Families: verification truncation, confidence truncation, block4/8/12/24/32, history lookup, adaptive block sizes, and two-branch target verification. A custom full-block no-op must exactly match original tokens and acceptance on eight64-token probes before hypotheses run. All20 hypotheses passed; this means execution/correctness gates, not scientific improvement. The best overall point was history suffix3/block32 at1.037× (interval0.956–1.178), with5/8 exact native outputs. Its two code requests showed1.332×, requiring immediate broader validation. Original block1 AR already differs numerically from block16; changed-length hypotheses must receive explicit quality/agreement follow-up, not an assumed guarantee.

**29192 completed**, immutable v16, `radical-followup.json`, four GPU lanes each bounded540seconds:

- lane0: test history suffix3/block32 and confidence0.3 on8 HumanEval requests, cap512, two timing repeats.
- lane1: history lookup suffix/block variants2/32,3/48,4/32 on4 requests per workload, cap256.
- lane2: branch only when the first draft token's top-two logit margin is below0.5/1/2, plus a four-branch variant covering alternate tokens at the first three proposal positions.
- lane3: recycle unused proposal tails after verification, with minimum tail4/8 and at most1/2 recycling rounds. Every reused proposal is verified by the target; pending target features are accumulated until the next drafter call so cache positions stay aligned.

The radical runner preserves per-variant failure traces and partial JSONL. Repeated screens require identical tokens/acceptance across timing repeats. `analyze_radical.py` clusters repeated timings by request and reports all workload regressions, not only the selected maximum. Eight local tests pass; GPU no-op/branch-prefix gates remain mandatory in every wave.

Reserved `data/confirmation.json` contains128 requests (32 per workload, indices32:64), disjoint from new `data/screening.json` (indices0:16) and all eight-request tuning screens so far. No globally unseen-data claim: records came from an older project manifest. Do not evaluate confirmation until a checkpoint and decoding settings are frozen.

## Next required implementation and experiments

1. Continue the rapid waves recorded below. Job29182 is canceled, not an active dependency. Preserve the at-least10% target and reject unreplicated pilot maxima.
2. Build clear seed/data/capacity plots from the collected authoritative summaries; avoid plotting different loss scales as though comparable. Three joint seeds and matched batch8 data points are now complete.
3. Breadth and repeated-output analysis completed in EVALUATIONS.md with request-clustered bootstrap intervals. Probe baseline AR differences using the same divergent prefixes to distinguish numerical shape effects from incorrect conditioning. Preserve partial JSONL if future evaluation times out.
4. Cover remaining objective/position weighting and three-tap/native additional-training controls with adaptive rejection reasons. Longer four-layer fits helped but remain substantially behind; avoid treating that as a universal impossibility. Native-KL vs target-KL, CE/hard/mixed, uniform/early weighting, and materially smaller interface/width designs remain candidates to investigate.
5. Repeat selected controls across at least three seeds, tune block sizes on development, then freeze checkpoint and settings before128-request/2,048-token repeated confirmation. Final report needs interpretable graphs and a bounded verdict; no supported faster-native result yet.
Offline token/prefix matches are diagnostics, not measured autoregressive speculative acceptance. Raw source solutions may be imperfect. Native-drafter KL conditions teacher and student on identical available prefix features and masks; target KL uses teacher-forced target next-token distributions. All new code and evidence stay in this separate experiment folder.

## Radical follow-up and prediction-error studies

The 29184 two-code-request history-lookup gain did **not** replicate: 29192 tested eight code requests, cap512, twice, and obtained1.000× [0.947,1.058]. Confidence truncation obtained0.999×. Other lookup variants reached at most1.010× overall; conditional branching0.974–0.994×; proposal-tail recycling0.749–0.853×. This rejects promotion of the earlier33% code pilot, not every possible retrieval policy.

**29203** (v17) completed12 draft-vocabulary/context-window policies in3:52. Shortlisting and bounded draft attention did not reach the target; long-cap top512 vocabulary retained0.9845× and window128 retained0.821×. The original full target verification remains unchanged.

**29204** (v18) completed four prefix-refiner fits,512 records/256 updates. The first native pass supplies a predicted prefix, and a trained second pass refines its tail. Full-depth prefix2/4 retained0.8610/0.8624× native throughput; compact three-layer refiners0.5938/0.7283×. Full models learned better predictions but did not offset the extra pass.

**29211** (v19) collected native rollout verification labels on64 Numina training questions and16 validation questions:2,435/535 blocks. Labels after the first rejection are masked because their conditioning prefix is hypothetical. Only actual native draft output features are inputs to the correction head; no target future feature is exposed.

**29228** (v20) trained four rank16/64 correction heads for512 updates. CE, error weighting, and a larger learning rate all failed to improve throughput: the best was0.9984×, the most aggressive0.9047×. The latter repaired88 validation first errors but broke301 previously correct positions. All16 timed outputs per arm matched native. **29235** (v21) tested first-position-only heads and margin losses with native-prediction preservation; ratios0.9823–0.9977×. These results motivate evaluating consecutive progress, rather than just the number of repaired errors.

**29248** (v22) collected64 code +64 general-instruction training questions and16+16 validation questions. Together with Numina, `progress-mixed.json` contains192 training /48 validation questions,9,311/2,318 blocks. Dataset revisions, licensing, exact and five-shingle overlap checks are recorded in the prompt manifest and attribution file. No global pretraining independence is claimed.

**29271** (v23) completed in11:46 using four GPUs. Its two fast head lanes finished in112/120 seconds: mixed CE rank16 retained0.99056×; margin+preservation rank64 retained0.99211×. Both consumed all192 questions. The two longer prefix-refiner lanes used2,048 records and2,048 updates (4,096 sampled blocks), completing in692/695 seconds. Prefix2 retained0.85661× (137.7 vs160.8 TPS); prefix4 retained0.87834× (145.0 vs165.1). Longer training improved validation prediction losses but still did not pay for the second pass.

Protocol change: from29271, post-EOS cached training positions are masked while retaining the EOS label;159 positions were removed from the mixed cache. Earlier29228/29235 throughput measurements remain valid, but their training masks differ. Mixed fits also use question-uniform sampling. See `PROGRESS_HEADS.md` for complete paired results and repair/break rates.

An attempted CPU-only cache staging to node06 (29236) was canceled when node07 became available; its single partial blob was quarantined by29241. Node06 has no usable model cache. These jobs allocated no GPUs and produced no model evidence.

**29282** (v25) completed in4:06: two soft-prefix-progress losses and eight stale-draft embedding hint strengths. Soft-progress temperature0.1/0.5 retained0.99261/0.99289×, each finishing in115 seconds. The surrogate rewards a differentiable consecutive prefix, not true acceptance. Hints blend prior unverified draft tokens into masked query embeddings without adding a draft pass; full target verification is retained. Strength0 passed exact token/acceptance control. Weak hints were near parity; stronger hints regressed sharply.

**29285** (v26) completed in2:57: six target-tail hint strengths plus six direct proposal-recycling policies. These use target predictions already computed beyond the previous rejection. Such predictions are hypotheses under an incorrect prefix, so every token is freshly target-verified. Cache features are accumulated until the next drafter call; no stale target cache is committed. No policy improved overall throughput; direct recycling sacrificed accepted progress despite skipping draft calls.

**29288** (v27) completed in1:56: eight lazy full-vocabulary target-head policies across block16/32 and chunk1/2/4/8/16. The target transformer verifies the complete block. Vocabulary projection is evaluated incrementally only until the first rejection plus its corrective token is known. Chunk16/block16 passed exact full-head control at0.9993×. Block16 chunk1/2/4/8 retained0.8540/0.9199/0.9729/0.9911×. Synchronization outweighed skipped computation. Small GEMM shape changes can cause numerical output differences, which are recorded explicitly.

Every wave uses at most four GPUs; every lane in29282/29285/29288 contains individual hypotheses finishing in minutes. There is still **no confirmed10% overall gain**. Nineteen local correctness tests pass as ofv30, including causal tail alignment, EOS masking, masked gradients, warm-start control, exact first-rejection/corrective-token handling, and isolation of the draft vocabulary head from target verification. Reserved128-request/2,048-token confirmation and selected-candidate seed checks remain unrun.

## Native-generated sequences and component profiling

**29289** (v28) had three successful lanes and one configuration failure. Math and code lanes each collected64 training/16 validation native greedy responses and freshly computed causal target features on those sequences, at256 generated tokens and512 total sequence tokens. All160 records were eligible. The instruction lane used an incorrect domain name and failed before collecting records; its source snapshot and failure are retained. The fourth lane profiled eight native development requests with CUDA events and exactly reproduced tokens and acceptance. No hidden state under a rejected hypothetical prefix is used as a correct sequence feature.

Profile: target verification occupied80.5% of instrumented wall time (75.8% backbone +4.7% vocabulary head); draft backbone12.3%, draft head4.7%, prefill1.2%, other1.2%. Event spans can include host-dispatch gaps. The head-only Amdahl estimate is just1.050× even if all target projection cost disappeared. `NATIVE_PROFILE.md` and its graph document the measurement. These diagnostics support focusing on accepted progress per target pass.

**29290** (v29) completed the instruction collection with the corrected source key:59/64 training responses had a complete answer block; all16 validation responses qualified. The merged on-policy index has187 training /48 validation questions, with source hashes, exclusions and causal gates. Three parallel native-generated math/code fits used128 training/32 validation records,512 updates and1,024 sampled blocks. All were evaluated on eight mixed-domain requests at512 output tokens. Full native target-KL retained0.99602×, full native sequence-CE0.99092×, and two-tap target-KL0.94453×. All outputs matched original native. Fits finished in295–300 seconds. The validation-best checkpoint was at256 updates; standard results explicitly evaluate the final512-update weights.

**29295** installed the private TorchAO0.15.0 dependency on a CPU-only allocation after the login-node installer hit a memory limit. Existing Python packages were not changed. Torch2.9.1/TorchAO0.15.0 matches the official compatibility table. The earlier attempted pip command had no pip module and installed nothing.

**29296** (v30) failed all four lanes before timing at quantized checkpoint reload: a generic `cuda` map_location produced a device mismatch against the tensor subclass's indexed `cuda:0` storage. Preserved source, logs and lane statuses. The fix specifies the indexed current CUDA device; no quantization arithmetic or acceptance policy changed.

**29297** (v31) completed all four int4 proposal screens with copy/reload/frozen-control gates passing. Group32 draft-only retained0.94728×; group32 draft+head0.94327×; group128 draft-only0.95254×; group128 draft+head0.96419×. All16 timed outputs per arm matched native. Group128 draft-only preserved mean accepted progress (4.522 vs4.520), so the eager runtime regression is not explained by lost acceptance. Original BF16 target/head remain untouched. Packing/setup is reported separately. This is standard quantization with prior art, not a new algorithm claim.

**29298** (v32) failed before timing because independently compiled bound layer methods exhausted the32-graph limit. The compiler's recorded guard failure was the function object's identity, not a mathematical or quantization error. The fix uses one shared compiled functional linear entry point with explicit weight/bias arguments. A regression test exercises40 independent linear modules through the shared function;20 local tests pass.

**29301** (v33) completed in2:14 with all gates passing. Shared compiled proposal linears retained0.8903× for BF16 draft-only,0.8290× for int4 draft-only,0.8987× for BF16 draft+head and0.8757× for int4 draft+head. The second timing repeat also regressed, so this is not solely first-repeat compilation. All16 timed outputs per arm matched native. The target remained BF16 and uncompiled. These results do not rule out other compiled kernels, but do not support promotion of this implementation.

## Midpoint prediction within one draft pass

Implemented `midpoint_conditioning.py`: a hook after a selected draft layer projects the next1–2 query states through the shared final norm and frozen target vocabulary head. It adds alpha times(predicted token embedding minus mask embedding) to those query slots. Remaining draft layers see the predicted prefix in the same forward pass. The known anchor and all other slots are untouched by injection. Inference uses hard argmax predictions; training uses an exactly equal hard forward with a soft-embedding straight-through gradient. An auxiliary intermediate CE loss accompanies the final target-KL loss. Target weights, embeddings and vocabulary head stay frozen.

This targets the second-pass refiner's overhead while adding a small intermediate head projection. All committed tokens still require the unchanged target verifier. Do not describe this as novel before comparing relevant prior work, or as positive evidence before full measurements. The hook shares the student's final norm; an independent intermediate norm is an unrun possible variant.

**29303** (v34) passed all four smoke lanes in46–49 seconds:16 training/4 validation questions,8 updates with accumulation2, four mixed-domain requests atcap128. Tests covered alpha0, prefix1 after two layers, and prefix2 after three layers atalpha0.25/0.5. All zero-injection, gradient/frozen-target and checkpoint-reload gates passed. Twenty-two local tests pass, including exact hard/straight-through forward identity with nonzero prediction gradients and frozen embeddings. Smoke results are engineering evidence, not sufficient training or a speed claim.

**29304** (v35) completed on four L40S GPUs:187 native-generated training questions,48 validation,512 updates with accumulation2, full five-tap/five-layer joint fitting. After layer3, compare prefix2/alpha0 auxiliary-only control, prefix2/alpha0.25, prefix2/alpha0.5, and prefix1/alpha0.5. All use final target KL plus auxiliary CE weight1. Each lane is bounded540 seconds and ends with eight mixed-domain requests atcap512, native/duplicate/student controls and checkpoint reload. All187 records must be consumed. These checkpoints need midpoint hooks reinstalled for inference; `run_lane.py` does so, but the general `evaluate_native.py` runner does not yet support them. Add that only if promotion is warranted. Reserved128-request/cap2048 confirmation remains unrun; no confirmed10% gain exists.


## Packed verification trees and direct prior art

29304 midpoint fits completed: auxiliary-only control1.0088×; prefix2/alpha0.25 0.9611×; prefix2/alpha0.5 0.9643×; prefix1/alpha0.5 0.9681×. No promotion.

**29307** (v36) screened packed ancestor-only trees in one target batch. Single-path control passed original token/acceptance equivalence. Full tree4 reached1.0881× on eight requests; tree8 only1.0188×. **29315** (v37) optimized cache compaction to copy only appended accepted nodes, preserving the old prefix. Tree4 breadth32×two repeats retained1.0558×. One-token leaves reached1.0933× in small pilots. **29337** (v38) varied branch extent and leaf ranks: top5 at the first four positions reached1.1577× in an eight-request pilot; tree5 suffix6 reached1.1352×. A broader tree4/suffix8 test returned1.0522×. These pilots are adaptively selected.

**29338** (v39) completed all four lanes in383 seconds or less. The selected top5/first4 leaf tree reached **144.0085 vs128.6229 TPS =1.11962×** on32 development requests, cap512, two timing repeats. Mean progress5.6688 vs4.7637. Exact native BF16 outputs14/32 unique requests. Tree5/suffix6 reached1.09458×, exact10/32. Strict BF16 accumulation did not eliminate numerical differences (leaf3/8 exact); its pilot ratio1.1347× is measured against the same strict-precision native baseline. Increasing leaf breadth beyond top5 produced smaller pilot gains (~1.108–1.118×). The32-request set includes the original eight pilot requests and24 additional requests. This is a development result, not final confirmation.

**Numerical diagnostic29337/lane0:** selected four divergent full-tree4 cases and captured the first differing target predictor at the same true prefix. BF16 replay exactly reproduced both saved decoder outputs; all four packed-tree predictors had tied top logits. Casting model parameters (both target and draft) toFP32 made native and packed-tree outputs exactly identical throughcap512 in all four selected cases. This supports shape/rounding sensitivity in those cases, not universal output equivalence or a quality-rate estimate. FP32 timing was not evaluated. Original model parameters/precision were not changed in the BF16 speed studies.

**Direct prior art discovered:** Ringel and Romano, *Accelerating Speculative Decoding with Block Diffusion Draft Trees*, arXiv2604.12989 (DDTree), already constructs trees from DFlash parallel predictions. We make **no novelty claim for this general idea**. A matched-runtime DDTree heap-builder baseline is now implemented with MIT attribution, pinned upstream commit c96427a185677bf4133ed865dd1626a5041aef9b. It is an adapted algorithm baseline, not full reproduction of upstream timing. It uses the same target/draft/SDPA/precision and shared packed verification/cache logic as our fixed trees. Budgets exclude the bonus root:31 DDTree nodes match the fixed leaf's32 total nodes. Exhaustive small-tree prefix-mass and ancestor-mask tests pass;28 local tests pass overall.

Next wave compares DDTree budgets15/31/47 on eight requests/two repeats and the selected leaf policy on eight requests atcap2048. Reserved128-request/cap2048 confirmation remains unrun. No final >=10% claim has been established.


**29350** (v40) completed all four lanes in89–114 seconds, all gates passed. Adapted DDTree budgets15/31/47 (16/32/48 total nodes) reached1.1452/1.2349/1.2827× on eight development requests, cap512, two repeats. These are existing-method gains, not our contribution. Budget47 native baseline ran~2.4% slower than other simultaneous lanes, so compare paired ratios cautiously and remeasure broader. Fixed top5/first4 leaf atcap2048 dropped to1.0764× (124.9 vs116.0 TPS), eight requests/one repeat; it does not meet the target at the requested output cap. The earlier32-request/cap512 leaf estimate has paired95% interval[1.0648,1.1678]; the additional24 requests retain1.1072× [1.0531,1.1585], so a>=10% lower bound is not established.

Next v41 wave broadens adapted DDTree31/47 to16 requests, two repeats, cap2048; tests budgets63/95; and screens a GPU-only restricted leaf selector. The latter ranks alternative leaves by primary-prefix mass times alternative probability, with prefix exponents0/0.5/1, and selects16 extra leaves from15 positions ×four alternative ranks. It uses fixed32-node tensor packing without a CPU heap. This is explicitly DDTree-inspired, restricted to a single off-primary leaf, and makes no novelty claim. A separate test independently checks selected probability scores and all ancestor paths.29 local tests pass.


**29356** (v41) completed. Atcap2048,16 development requests/two repeats, adapted DDTree31 reached1.1758× andDDTree47 reached1.2019× (150.4 vs125.1 TPS). Both have5/16 unique outputs exactly native BF16. The eight-request/cap512 budgets63/95 reached1.2824/1.2679×. GPU-only adaptive leaves at prefix exponents0/0.5/1 reached1.1587/1.1624/1.1610×; none beat the matched-node DDTree31 pilot1.2349×.

**29363** (v42) completed three compact-checkpoint comparisons with original DFlash and full-drafter DDTree47 as simultaneous rotated controls (eight requests,cap512,two repeats). Two-tap/five-layer retained1.1922× original but only0.9403× full DDTree. Four-layer drop3/drop2 retained1.0228/1.0138× original, and0.8077/0.7997× full DDTree. All checkpoint hashes, exact reload and native-equivalence gates passed. Thus tree verification recovers some compressed-drafter performance, but compression currently adds no gain. DDTree selection temperatures0.5/0.75/1.25/1.5 reached1.1690/1.2782/1.2262/1.1382× original in one-repeat pilots. None clearly exceeds unscaled DDTree47/63; no temperature promotion.

Next: coverage-aware joint fits. Add a hinge that encourages the frozen causal target's greedy token to enter the student top5, retaining the original target-KL loss. Hinge=max(0,kth-highest non-target logit - target-token logit +0.2), weighted across positions by the existinggamma7 weights. Coefficient1; a full-model coefficient0 KL control is included. This is a heuristic local rank objective, **not** true differentiable tree acceptance or a novelty claim. Two compact fits warm-start hashed prior joint checkpoints; full fits start the released drafter. All targets remain frozen. Eight-update smoke precedes512-update fits on187 training/48 validation native-generated questions. Decode evaluation compares original DFlash, duplicate original, full DDTree47, and the student with the same DDTree47. Final128-request confirmation remains reserved.


**29373** (v43) passed all four8-update coverage smoke lanes in67–77 seconds. Frozen controls, gradients, tree no-op and checkpoint reload gates passed. **29379**, using the same immutable validatedv43 source, is running the full512-update fits,187 training/48 validation records, four GPUs, each lane bounded540 seconds. Initial validation of the full native drafter places the causal target token in the top5 on~69.3% of held-out positions; this is teacher-forced marginal coverage, not tree acceptance.31 local tests pass.

An independent inference wave (`tree-horizon-and-long-breadth.json`,v44) follows29379: fast lanes compare native draft horizons8/12/24/32 with DDTree47; the other two lanes compare DDTree47/63 on32 development requests atcap2048, with two timing repeats. At most four GPUs can run because of the explicit afterany dependency. Reserved confirmation remains untouched.


The next inference wave is job **29381**, queued after29379, sourcev44. Added an explicit confirmation-protocol validator for future `run_radical.py` runs: reserved-manifest use requires phase=confirmation, a hashed frozen selection matching model/checkpoint/decoder-source/precision settings,128 unique question IDs, cap2048, and repeated timing. No frozen candidate or confirmation results have been created yet.32 local tests pass, including rejection of changed decoder code after selection. Future raw inference logs also include decoded completions outside the timed region to support quality review.


**29379** completed all four coverage fits in319–351 seconds. Final student/full-DDTree throughput: standard target-KL control0.999×; full coverage0.982×; two-tap coverage0.945×; four-layer coverage0.793×. Corresponding original-DFlash ratios1.261/1.243/1.192/1.030×. None adds benefit over the existing full DDTree method. `TREE_COVERAGE.md` reports the matched controls and common validationKL, avoiding comparisons of unlike summed objectives.

A further primary-literature check identified direct related approaches: DART, JetSpec, Weaver, PCTree andDARTree. See `NATIVE_PRIOR_ART.md`; these have not been reproduced locally. General conditional reranking and parallel tree training are not unclaimed new directions.

Next implemented wave: append one previously observed in-context continuation to DDTree, merging shared prefix nodes, versus full DDTree63; and test whether wider trees95/191 recover the compact five-/four-layer drafters. History comes only from already known input/output tokens, and every merged node receives fresh target verification. It adds at most15 nodes toDDTree47 (at most63 total nodes including root), so the referenceDDTree63 has a comparable64-node upper budget, not exactly equal dynamic work. This is a heuristic related to continuity-aware prior work, not a DART reproduction.33 local tests pass, including shared-prefix merging and ancestor masks. Sourcev45 adds the confirmation-phase guard and decoded-completion logs.


**29381** completed all four lanes,107–115 seconds for fast horizon screens and399 seconds for broader lanes. DDTree47 at32 requests/cap2048/two repeats reached155.1384 vs127.9128 TPS,1.212845× [paired95%1.1483,1.2697], exact9/32 native outputs. DDTree63 reached156.4825 vs128.3985 TPS,1.218726× [1.1563,1.2791], exact11/32. Additional24 non-pilot requests retain1.2095/1.2206×. These are robust development gains of an existing method; reserved confirmation and full task-quality audit remain outstanding. Horizons8/12/24/32 yielded1.0329/1.1713/1.2206/1.1804× in eight-request/cap512 screens, below theblock16 DDTree pilot.

**29384** is running the history/compact-tree wave on four GPUs, sourcev45. Two history lanes compare suffix2/3 and continuation8/15; two compact lanes compare tree budgets95/191. Every lane is bounded540 seconds and includes original DFlash plus fullDDTree63. No additional fitting occurs.


**29384** completed all eight hypotheses. History suffix2/3 and length8/15 achieved1.2495–1.2545× original, but only0.9752–0.9796× fullDDTree63. Wider compact trees did not recover the lost quality: two-tap budgets95/191 retained0.9092/0.9068× fullDDTree63; four-layer0.7830/0.7706×. No promotion.

Next implemented wave (`adaptive-tree-budget.json`,v46) chooses among15/31/63/95 non-root nodes each round. It builds the maximum95-node DDTree, then maximizes(1 + sum of selected factorized prefix probabilities)/(1 + relative_node_cost × node_count), testing relative costs0.001/0.002/0.004/0.008. This is a heuristic cost-aware policy: the probability sum is a surrogate, and the affine cost coefficients are tuning assumptions, not measured target acceptance or hardware latency. Every arm includes original DFlash and fullDDTree63; all tree-selection overhead is timed.34 local tests pass, including adaptive-prefix truncation equivalence for unique scores. No reserved confirmation has run.

**29391** is now the active four-GPU adaptive-budget wave, sourcev46, Slurm wall limit6 minutes. The goal remains active. Strongest broader result is existingDDTree63 at+21.9%; none of our completed extensions has improved on it.


## Frozen end-to-end confirmation

**29391** finished the adaptive-budget screen. Relative costs0.001/0.002/0.004/0.008 retained0.9995/0.9763/0.9753/0.9262× fullDDTree63. No adaptive policy adds a resolved gain.

**29404** passed the four-arm plus AR pipeline smoke on development requests in36–56 seconds per lane. Timed arms are originalDFlash, compact joint linear, the same compact joint checkpoint withDDTree47, and the released full drafter withDDTree63. Original block1 AR runs once per request outside repeated timing for output agreement and quality review. A login-node Torch import failed due to the login memory mapping limit; installed versions were instead read from package metadata and successful GPU-job provenance. No environment was modified to work around it.

Selection is frozen in `data/frozen-native-confirmation.json`, SHA256 **688c439bfe20978f5ea2155f33e955bb6ff36a2762cfb078f4a5b0d4cb779db5**. It fixes checkpoint29162/lane0, both candidate decoding paths, fullDDTree63 reference, source/launcher hashes, Torch2.9.1+cu128, Transformers5.3.0, BF16 settings, cap2048 and two timing repeats. The128 reserved requests are now being consumed strictly for confirmation, not further tuning. AR reference generation is once per request. Sourcev48 is immutable; local analysis lives in a separate subdirectory.

Campaign: sixteen four-GPU waves, two requests per GPU, rotating workloads across lanes. Jobs29406,29408–29422 are chained afterok, each with a10-minute Slurm limit and540-second lane timeout; later jobs cancel on invalid dependencies. All completed lanes must pass exact repeat/reload/source gates and AR completeness checks. `analysis/collect_confirmation.py` records authoritative job states without submitting retries; `analysis/analyze_confirmation.py` refuses final estimates until all128×4×2 timed generations and128 AR generations are present.

Quality uses the existing vendored Qwen2.5-Math parser/grader with source hashes and a private local dependency directory (`/tmp/native-quality-deps`: antlr4 runtime4.11.1, word2number1.1, SymPy1.12, mpmath1.3). HumanEval executes supplied tests under bubblewrap with no home/network visibility and CPU/memory/output bounds. Sandbox checks cover positive/negative code, math, runaway code and hidden home access. Dialogue has no ground-truth label and will not be given a fabricated accuracy score. No RelaySpec training/mapper component is imported, and no manuscript is changed.

Confirmation checkpoint:72/128 requests (36/64 lanes) have complete repeated four-arm timings and AR outputs. Jobs through29415 are complete;29416 is running, with29417–29422 pending on successful predecessors. All collected protocol/repeat/runtime gates pass. Quality scoring is incremental and content-cached, with no interim quality table or speed ranking exposed for tuning. The inference source still exactly matches the frozen source hashes.


## Completed frozen confirmation and final decision

All 16 waves (jobs 29406, 29408–29422) completed successfully: 128 requests, four timed methods, two repeats (1,024 timed generations), and 128 AR quality generations. Every lane passed source/runtime/reload/repeat gates. Four GPUs were used at most; the longest wave was 177 seconds. Total allocation wall time was 1,928 seconds (2.142 allocated GPU-hours). The user queue is empty.

Final throughput: original DFlash **169.5 TPS**; compact joint linear **159.2 TPS (−6.1%)**; compact joint + DDTree47 **185.9 TPS (+9.7%)**; full drafter + DDTree63 **199.3 TPS (+17.6%)**. Paired 95% throughput ratio intervals versus original are [0.930, 0.948], [1.070, 1.123], and [1.155, 1.198], respectively. Only full DDTree establishes the requested >=10% gain. DDTree is existing prior art. None of the new extensions beats it.

The compact linear path reproduces original DFlash outputs on all 128 questions, but its accepted progress is lower (5.944 vs 6.342) with almost unchanged complete round time (37.25 vs 37.34 ms). Full DDTree achieves 8.329 progress at 41.70 ms/round. These are amortized wall times, not isolated kernel costs. Exact AR output agreement is 40/128 for original and compact linear, 40/128 for compact-tree, and 42/128 for full-tree. All methods cap exactly one MATH request. Numerical equivalence and quality preservation are not assumed.

Quality v1 produced 137/SIGKILL process failures on valid math answers. Its source, protocol and raw scores are preserved in quality-v1. Quality v2 applies the same 20 CPU-second/30 wall-second sandbox limits to every output, treats resource failures as unscored, and rescored the entire campaign. All 640 outputs are covered, with no unscored math/code outputs; all six unique code failures were actual test assertions. Final GSM8K/MATH/HumanEval counts (each /32): original 28/30/31; compact linear 28/30/31; compact-tree 27/29/31; full-tree 29/28/30; AR 29/30/30. Dialogue has no ground-truth score and no capped responses. These samples do not establish quality equivalence.

The selected checkpoint was copied from temporary Turing scratch to local checkpoints/selected-29162-lane0.pt; SHA256 matches the frozen 83dd90d3590d27bca26245e49b9ae71133626df327b279ac062a16a86be226c5. The 34 local tests and revised sandbox checks pass; frozen inference/launcher hashes are unchanged. The final decision is in DECISION.md with three PNG/PDF comparison plots, complete raw outputs, quality audits and reconstruction scripts. This closes the bounded native joint-training comparison with a negative result for the proposed compact method, and a positive speed result for the existing DDTree baseline. The confirmation set is now consumed and must not be used for further tuning followed by a new claim of untouched confirmation.
