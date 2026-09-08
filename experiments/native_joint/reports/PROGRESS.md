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

**29192 is active**, immutable v16, `radical-followup.json`, four GPU lanes each bounded540seconds:

- lane0: test history suffix3/block32 and confidence0.3 on8 HumanEval requests, cap512, two timing repeats.
- lane1: history lookup suffix/block variants2/32,3/48,4/32 on4 requests per workload, cap256.
- lane2: branch only when the first draft token's top-two logit margin is below0.5/1/2, plus a four-branch variant covering alternate tokens at the first three proposal positions.
- lane3: recycle unused proposal tails after verification, with minimum tail4/8 and at most1/2 recycling rounds. Every reused proposal is verified by the target; pending target features are accumulated until the next drafter call so cache positions stay aligned.

The radical runner preserves per-variant failure traces and partial JSONL. Repeated screens require identical tokens/acceptance across timing repeats. `analyze_radical.py` clusters repeated timings by request and reports all workload regressions, not only the selected maximum. Eight local tests pass; GPU no-op/branch-prefix gates remain mandatory in every wave.

Reserved `data/confirmation.json` contains128 requests (32 per workload, indices32:64), disjoint from new `data/screening.json` (indices0:16) and all eight-request tuning screens so far. No globally unseen-data claim: records came from an older project manifest. Do not evaluate confirmation until a checkpoint and decoding settings are frozen.

## Next required implementation and experiments

1. Poll exact job29192 and collect all lane artifacts. Job29182 is canceled, not an active dependency. Continue short radical hypothesis waves based on observed results; preserve the at-least10% target.
2. Build clear seed/data/capacity plots from the collected authoritative summaries; avoid plotting different loss scales as though comparable. Three joint seeds and matched batch8 data points are now complete.
3. Breadth and repeated-output analysis completed in EVALUATIONS.md with request-clustered bootstrap intervals. Probe baseline AR differences using the same divergent prefixes to distinguish numerical shape effects from incorrect conditioning. Preserve partial JSONL if future evaluation times out.
4. Cover remaining objective/position weighting and three-tap/native additional-training controls with adaptive rejection reasons. Longer four-layer fits helped but remain substantially behind; avoid treating that as a universal impossibility. Native-KL vs target-KL, CE/hard/mixed, uniform/early weighting, and materially smaller interface/width designs remain candidates to investigate.
5. Repeat selected controls across at least three seeds, tune block sizes on development, then freeze checkpoint and settings before128-request/2,048-token repeated confirmation. Final report needs interpretable graphs and a bounded verdict; no supported faster-native result yet.
Offline token/prefix matches are diagnostics, not measured autoregressive speculative acceptance. Raw source solutions may be imperfect. Native-drafter KL conditions teacher and student on identical available prefix features and masks; target KL uses teacher-forced target next-token distributions. All new code and evidence stay in this separate experiment folder.
