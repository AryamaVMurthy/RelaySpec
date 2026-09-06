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

## Wave4 replicated screen and wave5/6 decisions

Wave4 job28400 passed all four lanes in227–318 seconds. On32 exposed questions with128-token caps, DFlash two-tap retained97.70% [95.69,99.60] of full-map throughput on GSM8K and101.60% [99.51,103.73] on MATH; trained factor1024 retained96.92% and100.79%, respectively. These intervals exclude fitting-seed and selection uncertainty. Two-tap uses40% of full mapper parameters; factor1024 uses45%. EAGLE two-tap retained96.27% [93.96,98.53]. Its identical-checkpoint control matched output hashes, acceptance trajectories and call counts on all32 requests; pooled progress is stable. The earlier apparent progress discrepancy arose from comparing pooled versus mean-per-request metrics.

Native rank1536 retains99.14% [96.93,101.54] of repacked-native throughput with45% of native projection parameters; rank1024 retains93.81% and rank2048 retains100.92%. These are projection-only reductions, not full-model reductions. The released native baseline remains present in the screen. Figure and input hashes: figures/interface_compression.pdf and compression-screen-summary.json.

Wave5 job28413 tests last-two-tap14B DFlash/EAGLE on16 exposed MATH requests plus second-seed8B DFlash/EAGLE fits on16 exposed GSM8K requests. Actual14B cache metadata fixes taps[28,37]. DFlash14B has no native baseline in the existing pinned config, explicitly omitted; AR and paired full-map remain. All lanes540-second caps.8B seed lanes passed in199 and241 seconds while14B evaluation continued.

Wave6 is a new fixed comparison, not optional extension of the earlier N512/N2048 comparison. Protocol configs/autoresearch/20260907/confirmation-v1/protocol.json freezes64 reserve questions, full-map and two-tap checkpoint hashes,2048-token cap and a95% throughput-retention lower-confidence-bound criterion. All four16-question shards must complete before inference. Answer quality and truncation are required; no1point quality noninferiority claim. Native rank1536 is preselected as a secondary comparison before any new confirmation output. Exposure audit scanned964 local rank files across both worktrees and found0/256 reserve IDs. Scope is collected local history, not global contamination assurance.

Prior-work reading refined direction choice: DeLS-Spec2607.07409 Sections4.1–4.3 already adds an independently trained local RNN and unigram-corrected logit fusion to frozen DFlash. Do not pitch such a local-correction head as a new idea. DFlare2606.02091 Sections1/3.1 already studies learnable layer fusion with deeper jointly trained drafters; our current layer-reduction evidence concerns retaining a released frozen drafter with a small fitted interface, not first use of layer selection/fusion.

## Fresh confirmation and activation-aware compression

Wave5 completed all four lanes in199–341 seconds.14B DFlash two-tap retains99.28% [95.03,103.76];14B EAGLE93.73% [90.74,97.25]. Second-seed8B DFlash97.63%, EAGLE94.77%. These are exposed16-request screens and do not prove tight cross-family retention.

Wave6 job28414 passed all four fresh16-request shards in75–102 seconds. Aggregated64-request primary analysis passes the frozen retention criterion: two-tap156.86 versus full160.67 tokens/s,97.627% [96.228,99.030]. Both62/64 correct, identical output hashes on64/64, no cap-length outputs, mean310.28 tokens. Conservative paired accuracy interval[-6.62,+6.62] percentage points, not a one-point noninferiority result. Wave7 job28416 confirms preselected native rank1536 descriptively on the same64 requests:176.25 versus released178.65 tokens/s,98.657% [97.583,99.752]. Repacked full native177.16 tokens/s (99.163%). All62/64, identical outputs and no cap-length outputs. No new sampling/tuning on these64 questions is allowed.

Scoring/auditing script scripts/analyze_research_confirmation.py verifies exact64 IDs, disjoint shards, methods, checkpoint hashes, model/generation settings and pinned scorer files. It uses the existing Qwen math scorer and conservative paired-accuracy helper. Rendered tables come from scripts/build_interface_confirmation_assets.py. Main paper includes the two-tap fixed result, appendix gives protocol, bounds, family limitations and native application with both controls. PDF compiles to39pages; conclusion remains onpage9. New appendix pages18–19 were rendered and inspected. Full manuscript audit passed all checks except language constraints (new semicolons, now corrected) and stale full-document visual review. Do not claim final submission audit passes until complete current-PDF visual review is recorded.

Wave8 job28419 passed all four lanes in114–163 seconds. The new diagnostic uses512 cached fitting records, up to8 tokens each, takes output PCA of XW^T, and deploys U(U^T W).128 separate fitting-validation records supply projection MSE. Exact cache file hashes and target taps are checked. Weight-only SVD is recomputed alongside at matching ranks. This is a known data-aware compression idea applied to the drafter interface, not a novel general compression algorithm (relevant prior art ASVD arXiv2312.05821; SVD-LLM arXiv2403.07378, primary abstracts verified2026-09-07).

On eight exposed128-token requests, activation/weight-only throughput retention: DFlash8B rank51272.4/47.6%, rank102495.8/93.6%; EAGLE8B rank51281.2/34.5%, rank102490.4/52.1%; DFlash14B rank51272.8/50.3%, rank102493.6/90.7%. Native DFlash rank51273.3/71.5%, rank102492.4/96.4%: activation awareness is not uniformly beneficial. Strongest follow-up is EAGLE activation-aware rank1024 on32 exposed questions with the trained factor1024 comparator, before any additional confirmation. Preserve negative native outcome.

Wave9 job28421 tests a different mechanism: early[1,9] versus spaced[9,25] two-tap fits in DFlash and EAGLE, matched to the existing late[25,33] mapper's parameter count and fitting budget. Four16-request GSM8K lanes,540-second cap. This tests whether late-layer information, rather than capacity alone, explains the observed two-tap sufficiency. Fresh64-question confirmation stays excluded from all these development screens.

## Equal-capacity depth mechanism and matched compression follow-up

Wave9 job28421 passed all lanes in203–257 seconds. Same16 GSM8K requests,128-token cap,512 fitting records,8192updates,seed1729: early[1,9] maps retain56.29% DFlash and56.95% EAGLE throughput; spaced[9,25] retains91.48% and91.44%. The exactly matching16-question subset from wave4's late[25,33] maps retains98.71% and96.19%. Each arm uses its own paired full mapper as baseline. This is evidence for depth-dependent information at fixed parameter count, not a universal optimal-layer claim. scripts/build_layer_depth_assets.py checks the shared request IDs and generates figures/layer_depth.pdf and layer-depth-summary.json. Added the figure and interpretation to the appendix.

Wave10 job28425 queued after28421 and now running: lane0 fits the previously absent EAGLE rank1024 direct-teacher comparator atN512/8192updates, then compares it with dense, activation-aware and weight-only maps on32 exposed GSM8K requests. Lane1 tests the two posthoc EAGLE rank1024 maps on32 exposed MATH requests. Lane2 compares both posthoc and direct-trained rank1024 DFlash14B maps on32 exposed MATH requests. Lane3 tests native DFlash2/4/8-bit weight rounding in BF16 on32 exposed GSM8K requests. No integer-kernel speed/memory claim. Every lane remains capped540seconds.

Added explicit ASVD and SVD-LLM citations to the native compression application. ASVD author list checked against arXiv2312.05821v5 (includes Dawei Yang); SVD-LLM authors and ICLR2025 venue checked against arXiv2403.07378v5 comments. OpenReview direct page presents a browser challenge, so no claim to have read its review discussion. The paper treats these as established compression precedents, not novel algorithms introduced here.

## Cross-domain depth application queued

Wave11 job28427 is dependency-chained after wave10. Four independent lanes evaluate early, spaced and late two-tap maps against full relay, AR and native references on code (MBPP,8requests,512-token cap) and dialogue (MT-Bench,8two-turn conversations,256tokens/turn), for both DFlash and EAGLE. These are previously exposed development workloads; no fresh64 confirmation request is used for tuning. Whole conversations form bootstrap clusters. These are mechanism screens, not code test-pass or dialogue-quality claims.

Wave10 direct EAGLE rank1024 training completed successfully:23,592,960 parameters,8192updates,512 distinct records,92.91seconds of training updates,124.95seconds total worker time excluding imports. This is a full-five-tap factorized comparator despite the inherited trial study label reduced_taps. Most remaining wave time is decoding, not mapper optimization. MATH32-request lane1 passed353.8seconds: activation-aware EAGLE rank1024 retains97.19% versus59.31% weight-only. DFlash14B lane2 passed353.4seconds: activation-aware96.59%, direct-trained97.28%, weight-only92.67%. Other lanes were still live at this update, so no whole-wave completion claim yet.

Main conclusion now highlights the fixed two-tap and native projection findings as applications/insights in addition to retargeting. The native comparison remains explicitly secondary, and existing nonlinear/regularization/adaptation comparisons remain discussed in the paper.

## Timeout recovery and native teacher application

Wave10 lane0 hit its540-second cap after saving191/192 planned measurements. Its completed fitted checkpoint is retained, but the partial decoding rows are not promoted to a completed comparison. Other three lanes passed. Native rounding retention versus repacked native:2bit22.96%,4bit90.30%,8bit100.35%; all executed BF16, so this is precision sensitivity only.

Wave12 job28428 is dependency-chained after28427. Lane0 reruns the entire32-question EAGLE comparison using the completed checkpoint without refitting, preserving methods and caps. Lane1 adds that directly fitted rank1024 comparator to the MATH comparison. Lanes2/3 fit native DFlash interfaces from two late layers[25,33] or one[33],N512/8192updates,8 exposed GSM8K requests. Supervision is the released native fc plus hidden_norm applied to the exact cached full target features, not the old source-drafter y values. The teacher export records target/proposer IDs, taps, normalization and a checked hash. The trainer rejects mismatched cached target identity/taps and runs its batch-gradient equivalence pilot before fitting. This is distillation of a released native interface with a frozen drafter, not a new general compression algorithm.

Native-teacher implementation is in scripts/run_research_lane.py and scripts/fit_cached_mappers.py, source356b186.13 relay/transform tests passed; the actual native pilot and fits remain to be verified in28428. Every lane retains540-second cap.

Wave11 code lanes passed: DFlash early40.5%,spaced84.8%,late99.7% of full-map throughput; EAGLE early52.0%,spaced84.9%,late85.4%. These are eight exposed MBPP requests with512-token caps, not code test-pass evidence. The EAGLE result is a useful boundary: late-layer compression loses more on code than on the mathematical screen. Dialogue lanes remained live at this update.

scripts/build_autoresearch_registry.py now records every collected terminal lane, including failures and negative arms, in EXPERIMENTS.md and experiment-registry.json. Missing lanes are not assumed complete, and per-shard fixed-confirmation summaries are explicitly not substitutes for the64-request aggregate analyses.

## Completed cross-domain and native teacher evidence

Wave11 completed all four lanes. MT-Bench has16 turns clustered into8 conversations. DFlash early65.8%,spaced91.3%,late97.8%; EAGLE early69.6%,spaced93.0%,late91.9%. The paper now includes the code/dialogue table and explicitly states the EAGLE boundary and lack of quality-score confirmation. scripts/build_crossdomain_depth_assets.py retains per-method intervals in crossdomain-depth-summary.json.

Fixed64-request profile attribution (scripts/build_interface_profile_assets.py): full mapper prefill+decode is0.7446% of total request time, target verification81.9376%; two-tap mapper0.5819%, verification82.0440%. Target calls rise3391→3478. This explains why reducing mapper computation does not compensate for a modest acceptance loss. Added to appendix with measurement scope; not an isolated kernel/memory claim.

Wave12 job28428 completed all four lanes. EAGLE evaluation-only retry passed416.4seconds with all192 rows, and MATH passed399.7seconds. GSM8K32: direct-trained rank102496.07%, activation-aware93.86%, weight-only54.15%. MATH32: direct-trained96.48%, activation-aware96.73%, weight-only59.41%. Direct fitting remains useful; activation-aware compression is a strong posthoc alternative on some workloads, not universally superior. The posthoc path inherits dense fitting cost, so these are not equal-total-adaptation-budget comparisons.

Native teacher integration succeeded, including the actual batch-equivalence pilots. Eight-question native two-tap retains97.29% versus repacked native (98.2% versus released native), one-tap84.21% versus repacked (85.2% versus released). These are initial exposed screens. Wave13 job28429 now compares both students with released/repacked native and frozen rank1536 SVD across32 GSM8K,32 MATH,8MBPP and8two-turn MT-Bench cases, excluding all64 fixed-confirmation requests. FourGPU lanes, each540-second cap.

Paper depth plot now uses distinct circle/square markers to preserve family distinction in grayscale. Paper still compiles; full current-PDF visual-review record remains outstanding.
