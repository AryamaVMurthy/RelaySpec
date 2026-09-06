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

## Native operating points and calibration composition

Wave13 job28429 passed all four workload screens in95–181seconds. Relative to released native (recomputed paired reference, not dividing CI endpoints), two-layer students retain97.7% GSM8K,97.1% MATH,97.2% code and97.9% dialogue. One layer retains84.7%,88.8%,71.6%,87.0%. Rank1536 SVD retains98.8%,98.1%,98.6%,99.2%, with37.75M parameters versus33.55M two-layer and16.78M one-layer, compared with83.89M native. This is a parameter/throughput tradeoff, not student superiority. Added native-student table and method/scope to appendix through scripts/build_native_student_assets.py.

Added the completed32-request matched-rank comparison table through scripts/build_compression_objective_assets.py. EAGLE direct versus posthoc paths have equal deployed rank but different total adaptation budgets, since posthoc compression inherits dense fitting. Inputs X in the output-subspace method include mapper input normalization where required, and its subspace concerns linear projection outputs. No new-general-compression-method claim.

Wave14 job28430 fits four compact interfaces on512 records: retargeted/native DFlash × math-only/mixed math+general-instruction calibration, all8192updates and taps[25,33], evaluating32 exposed MBPP requests at128-token cap. Existing exact-cache pilots28132/28133 authorize the verified composition caches27959/27966. Cache-index prefixes were inspected: math512; mixed256math+256general_instruction; validation64+64. All four fit-complete records pass. Code decoding was still running at this update.

Wave15 job28431 is dependency-chained after28430 and reuses those exact four completed checkpoints for MATH32 and MT-Bench8two-turn screens. It was declared before interpreting code outcomes, so code gains will be checked for math/dialogue tradeoffs. No new data extraction or large-data scaling. FourGPU total,540seconds per lane.

## Paired compact composition and seed replication

Waves14–16 completed. Paired mixed/math retention (seed1729), with 95% request/conversation bootstrap: retargeted MATH101.11%[96.75,105.53], dialogue110.79%[107.76,112.54], code102.14%[98.85,105.55], GSM8K99.31%[96.79,101.88]. Native MATH101.45%[99.40,103.40], dialogue101.10%[98.84,102.74], code100.62%[98.59,102.70], GSM8K101.25%[99.65,102.97]. All paired outputs match within these capped screens. Only retargeted dialogue excludes parity, with eight conversations and one seed. Other differences should not be pitched as established improvements. Artifact builder build_compact_composition_assets.py records raw hashes and paired intervals, and adds the complete table to the appendix.

Wave17 job28433, source6b25ccb, is running four fits at seed1730, retargeted/native × math/mixed, each with original-seed paired controls on the original eight dialogue conversations. All lanes retain540-second limits. Wave18 is prepared to compare all four maps within each worker across sixteen additional dialogue conversations in two disjoint eight-conversation shards per interface family. These are broader development screens, not an untouched confirmation set. The original eight conversation IDs are excluded explicitly; selection is frozen before seed1730 decoding finishes.

Wave18 submitted as job28434, source3b6d3ed, dependencyafterany28433. It cannot overlap the fourGPU allocation. New compact-composition appendix builds successfully in the41-page PDF; full visual review remains outstanding.

Wave17 completed all four lanes in272–291seconds. Relative to the paired math seed1729 map, retargeted math seed1730 retains99.59% and mixed seed1730 retains111.12%[108.21,113.23]. The old mixed seed1729 paired control repeats110.77% in that same worker. Native seed1730 differences remain small. These compare each new fit to the old seed in the same worker, not a directly paired seed1730 mixed/math ratio; wave18 provides that pairing on sixteen additional dialogue conversations.

Wave19 job28435 (source817c20c), dependencyafterany28434, tests early-layer nonlinearity. Four lanes fit DFlash/EAGLE × factorized-linear/GELU-MLP at width1950, exactly20,966,400 parameters, layers[1,9], N512/8192updates/seed1729. Existing early-dense and late-dense20,971,520-parameter controls are paired in each eight-question128-token GSM8K screen. A negative result would limit the tested nonlinear fitting approach, not prove missing information is unrecoverable. Every lane keeps540-second timeout.

Wave18 job28434 completed all four lanes with80/80 rows each. Sixteen additional conversations (32 turns) were combined only after checking shard IDs, initial-screen exclusion, complete method pairing and identical checkpoint hashes across shards. Retargeted mixed/math throughput: seed1729 107.50%[104.34,110.44], seed1730 107.52%[104.44,110.21]. Native:101.09%[100.12,102.08] and100.28%[99.47,101.14]. All paired turn outputs match. Same conversations across seeds, hence not independent benchmark samples. Analysis and fitting-domain hashes are in analyze_dialogue_composition_extension.py and dialogue-composition-extension-summary.json.

The appendix now includes both seed extension estimates and validation diagnostics. General-instruction errors fall0.445→0.289 for retargeting and0.0468→0.0298 for native, while math error increases. Similar proportional feature-error improvements produce different throughput changes. The main composition paragraph includes the replicated7.5% versus0.3–1.1% contrast. These remain development/capped results without dialogue-quality scoring.

The reserve statement now correctly says the original256-question confirmation left the reserve unused, and the later separately frozen interface study consumed64 reserve requests. Main text remains within nine pages at the last successful audit. Generated evidence, citations, source anonymity and PDF parsing pass. Full current-PDF color/grayscale visual review remains outstanding.

## Early-layer nonlinearity and downward retargeting

Wave19 job28435 passed all four lanes in139–180seconds. At exactly20,966,400 parameters, DFlash early factorized/MLP maps retain56.0%/50.0% full throughput, with training errors0.333/0.271 and validation0.409/0.439. EAGLE retains57.9%/57.4%, training0.333/0.272, validation0.411/0.399. Paired late-dense controls retain97.4–98.6% DFlash and92.6–92.7% EAGLE on these eight exposed questions. Better training fit does not close the depth gap. Added full four-arm table and measured limitations to appendix via build_early_nonlinearity_assets.py.

Wave20 job28440 is a four-GPU end-to-end boundary pilot, not four independent research lanes. It retargets the4B DFlash drafter to Qwen3-0.6B revisionc1899de289a04d12100db370d81485cdf75e47ca, layers[1,7,13,19,25]. Cached source/target tokenizer.json and tokenizer_config.json hashes are byte-identical, recorded in downscale-pilot-protocol.json. All cache/gradient/checkpoint/duplicate-map gates passed. Its64records/16updates are pipeline validation, not an achievable-throughput claim.

Wave21 job28448 extracts512 fitting and128 validation records for that exact configuration after the pilot. The remote extraction-complete gate passes. Both stages have540-second process limits and fourGPU allocations. Pipeline jobs use pilot-gate/extraction-complete files rather than lane-status files; missing lane-status is expected for these explicitly typed ledger entries. Next comes the existing exact-cache capacity pilot before full fits, preserving the repository's cache-specific gate requirement.

The512+128-record 0.6B cache extraction took32.77seconds including model load, producing1.85GB with index SHA256 ee1c15bbdd3cf6dee4390b67f38f266ad1fbdbb8463645c1477881e863cb0d73. Wave22 job28455 runs the existing exact-cache capacity pilot: four16-update fits, gradient-equivalence checks, decoding and duplicate-map equivalence. Planned full trials are two dense seeds1729/1730, factorized1024 and MLP1024, each512records/8192updates. Final throughput will be judged against AR and source reuse, not the undertrained pilot maps. The registry now distinguishes stage-gated fourGPU pipeline jobs from independent research lanes.

Wave22 exact-cache capacity pilot passed. Wave23 job28457 now fits all four declared maps for8192 updates on512 records using that exact-cache gate. The next evaluation config was frozen before fit outcomes:16 exposed GSM8K requests,512-token cap, AR and optimized source reuse, both dense seeds, factorized1024, MLP1024, and a duplicate dense-map control. It measures deployment usefulness against AR rather than improvements over the16-update pilot. All stages retain four GPUs and540-second timeouts.

Wave23 passed all four full fits. Worker totals54.3–61.3seconds. Dense has13.1072M parameters and validation error0.3159 at both seeds, factorized1024 has7.86432M/error0.3211, matched MLP/error0.3297. Wave24 job28458 now runs the frozen16-question512-token evaluation after the completed fit gate. No throughput inference from fitting error alone.

## Downward retargeting and numerical agreement

Wave24 job28458 passed all112 measurements and duplicate-map trajectory checks. On16 GSM8K requests, AR51.55tokens/s, source reuse102.63, dense seed1729 167.97, dense seed1730 168.22, factorized1024 161.19, MLP1024 147.93. Dense/AR throughput3.258[3.002,3.496] and AR/dense request time3.317[3.067,3.548]. No method hits512-token cap. AR scores9/16, all speculative methods10/16. Exact AR matches only7/16. Conservative paired accuracy-difference CI[-23.9,33.8]pp does not prove superiority or noninferiority. Source verification consumes46.8% of source-reuse time; dense eliminates it while progress/cycle drops5.45→4.61. Analysis validates configurations, IDs, checkpoint hashes, and pinned scorer. Added complete seven-method table and limitations to appendix.

Wave25 job28459 passed all four lanes in61–78seconds. Same sixteen questions and BF16-fitted dense checkpoint, separate AR comparisons in BF16 and full FP32. Every model and mapper changes precision together, with TF32 disabled for FP32. BF16 repeats7/16 exact matches, FP32 gives16/16. Cross-precision AR agreement is only5/16 (and relay agreement5/16). This supports numerical effects in the sample but changes kernel dispatch too and does not isolate target precision or prove agreement for all prompts. DFlash benchmark now supports explicit precision with unchanged BF16 default. Ruff and10 relay tests pass; actual FP32 GPU runs pass. Evidence is in analyze_precision_screen.py/precision-summary.json and is added to the manuscript.

Wave26 is prepared to repeat AR/relay comparisons on eight code requests and eight two-turn dialogue conversations, each in BF16 and FP32. No fitting or candidate selection on these outputs, and no code-test/dialogue-quality claim. The aim is applicability and numerical reliability, not another mapper-size sweep.

Wave26 submitted as job28460, source61c1e3b, four independent code/dialogue × BF16/FP32 lanes with540-second caps. The precision finding is now mentioned in main quality discussion. Wording was tightened after the initial addition pushed the main text to page10; the rebuilt main-text boundary is verified back on page9, with no undefined references or overfull boxes in the final LaTeX log. Full-PDF visual review is still outstanding.

Wave26 job28460 passed all lanes in59–82seconds. Small-verifier code throughput/AR2.52BF16 and2.79FP32, exact matches5/8→8/8. Dialogue throughput/AR1.94BF16 and1.95FP32, matches8/16→16/16 turns across8conversations. Added the six-row GSM/code/dialogue precision table with paired request/conversation intervals via build_precision_breadth_assets.py. Neither code execution nor dialogue quality is scored.

Wave27 job28464 passed all lanes: BF16 workers87–104seconds, FP32 workers152–179seconds. Main8B verifier on the same16GSM8K questions gives12/16 BF16 and16/16 FP32 exact AR matches; no answers capped. Actual dtype metadata verifies target/drafter/mapper FP32 and TF32 disabled in FP32 arms. Source-trunk release now happens before target allocation when the load plan permits it, reducing transient loading memory without removing retained embedding/head. Model-load plan rejects source reuse with trunk removal.27 existing benchmark/relay tests pass; actual BF16/FP32 GPU runs pass.

The generalized analyze_precision_screen.py audits declared checkpoint hashes and observed parameter dtypes when recorded. precision8b-summary.json contains the new result. Main quality discussion now cites the8B diagnostic; the0.6B result remains in the appendix. Added Yuan et al., arXiv2506.09501 (2025), after reading primary abstract and LayerCast section. Numerical sensitivity/FP32 mitigation are established prior work, so these are RelaySpec diagnostics, not novelty claims. HEAL2606.21023 abstract was also inspected as a related numerical-mitigation lead, not implemented or cited as a benchmark competitor.

Wave28 is prepared to isolate the target output-head precision:0.6B/8B × normalBF16/FP32head, same16questions, AR and frozen RelaySpec in each arm. The helper copies a tied head before promotion so input embeddings remainBF16, then casts head inputs for FP32 projection. Two new tests exercise tied-embedding preservation and unsupported-head rejection; combined29 tests pass. No claim that this is a new precision algorithm.

Wave28 submitted as job28488, source242ac8a, after all previous jobs completed. FourGPU total,540-second lane caps. Updated manuscript compiles without undefined references/overfull boxes and main text ends on page9. Bibliography now includes the verified numerical-reproducibility prior. Full current-PDF visual review remains pending.

Wave28 job28488 passed all lanes in107–174seconds. On sixteen GSM8K questions, head-only FP32 changes own-runtime AR agreement7/16→9/16 for0.6B and12/16→16/16 for8B. It matches full-FP32 AR on only7/16 and13/16, respectively. The table explicitly separates own-runtime agreement from full-FP32 equivalence, and does not treat cross-mode rates as isolated kernel overhead. Actual metadata confirms BF16 embeddings/mapper and mixed BF16/FP32 target parameters in promoted arms. Added head_precision_table.tex and head-precision-summary.json through build_head_precision_assets.py.

Wave29 job28502, sourceb7a9fe8, now tests32 additional GSM8K questions and16 MATH questions, each in BF16/default-head and BF16/FP32-head execution, with AR, released native DFlash and frozen RelaySpec. The GSM8K set excludes the initial16. These remain development screens. Native DFlash also uses target.lm_head for draft logits, verified in the released source model.py:112, so head promotion affects that native proposal projection too. Four GPUs,540-second lane caps. RESEARCH_DECISIONS.md now distinguishes main confirmed findings, replicated development insights, and supporting numerical diagnostics; no indefinite precision sweep is planned.


Wave29 job28502 passed all four lanes in297–391 seconds. On32 additional GSM8K requests head-only FP32 improves own-runtime AR agreement15/32→20/32; on16 MATH requests3/16→5/16. Native DFlash and RelaySpec output hashes agree on every request within all four settings. MATH is heavily capped (AR11/16, speculative11/16 BF16 and10/16 head-FP32), so these are sequence diagnostics, not full-answer quality. Added head_precision_extension_table.tex and the negative extension to the appendix. This closes head-only precision as a general repair direction.

Wave30 job28520 passed all four lanes in28–38 seconds. Full and two-tap frozen mappers were compared with exactly the same checkpoints under full hidden-state return and selected-layer capture. Two reversed-order repetitions of one synthetic prompt per length matched outputs, token counts, acceptance trajectories and target/draft calls exactly. Actual inputs2273 and8993tokens. At8993tokens selected capture saves2.03GiB full mapper and2.20GiB two-tap peak allocated memory. Both mapper copies reside in memory for both methods. This is engineering memory evidence, not task-quality or a new compression algorithm.

Wave31 job28521 passed all four lanes in49–77seconds. Actual synthetic inputs16826 and31526tokens. At31526tokens full mapper peak32.98→25.87GiB; two-tap32.73→25.03GiB, with exact decoding trajectories in both reversed-order repetitions. Runtime is effectively unchanged. No inference from two repetitions as independent requests.

Wave32 job28522, source970872c, tests the same long-input pairs after both methods immediately release prefill outputs/concatenated taps after conditioning. This controls unnecessary prefill-buffer lifetime before attributing memory savings to selective capture. Four GPUs,540-second caps;30 selected-tap/relay/benchmark unit tests passed before launch. Default execution remains unchanged outside the explicitly configured diagnostic.

Wave32 job28522 passed all four lanes in48–75seconds. Releasing prefill buffers immediately in both arms leaves peak savings unchanged:7.11GiB for five taps and7.70GiB for two taps at31526tokens. Request-time ratios0.994–1.002 across paired repeats. Added all12 rows, scoped methodology and limitations to the memory appendix via build_capture_memory_assets.py. Wave33 job28523, sourcef53d48a, checks exact paired trajectories on eight code requests and eight two-turn dialogue conversations with full/two-tap maps, two repetitions each. FourGPU maximum and540-second caps remain enforced.

Wave33 job28523 passed all four lanes in86–160seconds. Full/two-tap mapper × code/dialogue, two repetitions each, all exact trajectory controls pass (16paired code trajectories per mapper,32 dialogue-turn trajectories per mapper). Added a scope-limited appendix sentence and raw-hash capture-workload-summary.json.

Wave34 job28524, source8484a42, freezes a new native two-layer confirmation on64 previously unused reserve questions (sorted positions64:128), all four16-question shards. Scan of932 prior local report JSONL files found no selected IDs; this is not global/pretraining non-exposure. Primary: two-tap versus released-native throughput lower95% CI >=95%. One-tap and SVD1536 are descriptive secondary arms. Full repacked baseline retained. All checkpoint hashes were measured and committed before decoding. No optional extension or sample removal;2048output cap,540s lane cap, fourGPU maximum. Generalized existing confirmation analyzer accepts protocol path and frozen reserve boundaries while preserving prior defaults.

Wave34 job28524 completed all four lanes in168–178seconds (Slurm3m7s). All320 outputs passed frozen-ID/settings/checkpoint/scorer auditing. Primary two-layer student retains97.0535%[95.9334,98.2051]% released-native throughput, passing the fixed95% lower-bound criterion with60% fewer projection parameters. One-layer84.5517%[83.1066,86.1012]%; SVD153698.0828%[97.1338,99.0554]%; repacked99.2304%. All methods60/64correct, all64output hashes match released native, no2048-cap outputs. Two-layer paired accuracy difference0pp with conservative[-6.62,+6.62]pp interval. Added all five arms to native_student_confirmation_table.tex and the manuscript; conclusion now highlights the native two-layer application.128of256reserve questions have now been consumed by two disjoint frozen research sets. No optional sample extension.

Wave35 job28525, sourcea5e5e0c, tests depth in downward retargeting: same exact-pilot-gated0.6B feature cache,512records8192updates seed1729, early[1,7],spaced[7,19],late[19,25],last[25]. Three two-layer maps have matched capacity; one-layer has half. Each lane evaluates8 exposed GSM8K requests at512cap with its paired frozen full mapper. Four GPUs,540-second caps, no claim of fresh confirmation. Native-student manuscript addition compiled cleanly, main text remains9pages.

Wave35 job28525 completed in1m41s. Early/spaced/late two-layer maps retain52.5/79.9/82.2% of paired full mapper; one-layer60.7%. Validation errors.459/.374/.351/.430. Each two-layer map5242880params, one-layer2621440. Added downscale_depth_table.tex and explicit smaller-verifier limitation to the appendix. Native-student table page23 was visually inspected (not a full45-page review). Tightened native confirmation analyzer to require exact candidate set/hashes against the frozen protocol; reanalysis passed unchanged. Wave36 prepares matched-width1137 MLP/factorized late maps at seeds1729/1730 to distinguish linear-fit limitations from selected-feature limitations in this screen.

Wave36 submitted as28526, sourceab66d81. Four GPUs; matched MLP/factorized1137 at two seeds, each540-second cap. Updated paper compiles without overfull boxes/undefined references, main text remains9pages.

Wave36 job28526 completed in1m44s, all lanes pass. MLP seeds1729/1730 retain81.1/83.1% full throughput; factorized84.2/83.1%; paired late-dense82.8–84.3%. All fitted maps5239296params. MLP trainerror.304/.303, validation.345 both, versus late-dense prior.332/.351; improved reconstruction does not restore decoding. Added all four arms to downscale_nonlinearity_table.tex and closed this fitting direction. Main experiments now jointly report independently frozen native and retargeted two-layer retention with disjoint64-question sets.

Manuscript audit coverage improvement: the previous generated-assets check omitted autoresearch builders while saying all assets passed. Added scripts/audit_autoresearch_assets.py, which copies reports/configs/scripts into an isolated temporary directory and reruns17 builders without mutating source artifacts. All17 resulting manuscript tables/raster plots reproduce byte-for-byte. PDF metadata/provenance absolute paths are explicitly outside this byte check. Integrated the check into the main audit and narrowed its wording to registered coverage. Full audit rerunning; full current-PDF color/grayscale visual review still outstanding.

Expanded full manuscript audit passes every automated check, including17 isolated autoresearch asset reproductions and46 resolved citations; only complete matching visual review remains. Rendered all45pages in color/grayscale at1400px maximum dimension. Main pages1–9 reviewed in color and grayscale (pages1,3,5,9 have pixel-identical RGB-converted grayscale renders); no clipping/overlap/legibility problems observed. Progress and all90render hashes saved in visual-review-progress.json, complete=false. Appendix/reference pages10–45 remain; old full-review JSON is not updated or falsely reused.

PDF review extended through page18. Found and fixed two prose rounding errors: historical continuous endpoint5.02458969 is5.02(not5.03); EAGLE weight-SVD54.1476096% is54.1%(not54.2). Corresponding plots/tables were correct. Rebuilt all45pages and color/grayscale renders; only pages17–18 changed, reinspected both. All other rendering bytes match, preserving earlier reviewed pages. visual-review-progress.json now covers1–18;19–45 remain. Final full audit should follow completed visual review.

Visual review now covers pages1–27 in both color/grayscale, with unchanged PDF hash. Numerical diagnostics, native confirmation, small-verifier counterexamples and regularization/capacity tables inspected; no new discrepancy or layout defect found. Pages28–45 remain.

Visual review covers pages1–36 in color/grayscale;37–45 remain. Review identified a substantive deferred experiment in the manuscript: EAGLE14B lower-rate pilots passed but full8192-update fits were not run during the old timebox. Assess those exact-cache pilots/configs after the current PDF review for a bounded learning-rate follow-up, rather than extending closed nonlinear width screens.
