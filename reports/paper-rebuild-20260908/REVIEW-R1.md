# Independent scientific review, round 1

Reviewed on 8 September 2026. Scope: the current `paper/iclr2027/relayspec_iclr2027.tex`, current generated main tables, the complete family-scaling reports, the 128-question family evaluation protocol and status, and the full transfer reproduction report, executed source and paired statistics. This review does not treat the older manuscript QA PASS as scientific validation. No TeX was edited by this reviewer.

## Assessment and central story

The strongest paper is about **retargeting the conditioning interface of an inherited feature-conditioned drafter while retaining its prediction network and target-controlled verification**. Its practical value is amortizing drafter training and removing the source transformer. The scientific findings concern the amount and type of calibration that preserve useful proposals, and the limits of feature-fitting proxies. A linear map is credible because the experiments test alternatives and decoding, not because linear alignment itself is new.

The new results materially improve the paper. They establish a non-Qwen replication, a different-tokenizer transfer, a precise token-agreement check on 128 requests, and a richer calibration recipe that slightly exceeds a native drafter in a separate vLLM experiment. They also invalidate a uniformly small-data or uniformly best-objective narrative. The best story has two measured operating regimes: inexpensive context regression often suffices, while richer rollout-based layer/context fitting can recover native-level performance in a separate setting. Neither regime establishes a universal recipe ranking.

## Priority 0: contradictions and claim validity to fix in the rewrite

### 1. Scope is stale throughout the current manuscript

The abstract, related-work distinction from TriSpec and conclusion repeatedly restrict the method to a larger verifier. The conclusion says all tests use shared-vocabulary Qwen3. The new Llama-3.1-8B drafter to Llama-3.2-3B target goes toward a smaller verifier, and Qwen3-4B drafter to Llama-3.1-8B target uses different tokenizers. These are central evidence, not miscellaneous appendix notes.

**Repair:** define retargeting to a *new* target. Put a compact transfer table and a brief heterogeneous-tokenizer method paragraph in the main paper. Qualify which results cover two drafter architectures versus two language-model families. Do not call DFlash versus EAGLE-3 a model-family generalization test when the intended meaning is Qwen versus Llama. Source-to-target arrows must name the inherited drafter and actual target size.

### 2. Three timing protocols must remain visibly separate

The primary manuscript metric includes prefill. The new family report includes both request TPS and decode-only TPS. Its main-paper-compatible numbers are:

| Pair | AR request TPS | Old request TPS | Selected request TPS | Selected/AR |
|---|---:|---:|---:|---:|
| Llama-3.1-8B DFlash to Llama-3.2-3B | 45.73097 | 114.93266 | 115.44486 | 2.5244 |
| Qwen3-4B DFlash to Llama-3.1-8B | 21.60950 | 51.97266 | 59.41269 | 2.7494 |

The 116.52/59.83 TPS and 2.54/2.76 ratios are decode-only values. They must not be silently inserted into a table labeled end-to-end or request throughput. These pairs use FP32 target/mapper/head, TF32 disabled, BF16 frozen drafter/interface; the historical Qwen suite uses another precision/runtime. FP32 AR is a correct matched baseline, but its speedup does not establish the same speedup over BF16 AR.

The rollout-reproduction result is a third protocol: vLLM 0.28, BF16, batch-invariant execution, no-thinking greedy decoding, prefix cache off. Its generation-wall timing includes prefill, model setup and four warmups are excluded, and compile-affected requests are repeated with cold times preserved. Do not splice its 6.44x AR speedup into the historical MATH-500 range or compare its raw TPS to the other engines as an algorithm ranking.

### 3. The strong rollout result evaluates custom Numina prompts, not MATH-500

`reports/transfer-reproduction-20260907/full16384-results/source/src/prepare.py` constructs 16,384 training, 256 development and 4,096 evaluation records from the first pinned NuminaMath shard, after numeric/punctuation template-group deduplication. It evaluates the first 128 evaluation records. This is a custom held-out Numina split, with no semantic-independence guarantee.

**Repair:** name the dataset explicitly in every table/caption. Report mapped 167.83/167.97 TPS and native 163.57/163.67 TPS across the two speculative runs, 1.0260/1.0263 native ratios, and finite 128/128 token agreement. AR was measured once and reused for the reversed-GPU speculative repeat; do not describe two independent AR runs. Native parity is a stronger and less brittle conclusion than universal native superiority.

### 4. The new comparisons do not isolate the loss or data count

The old 4,096-record and new 8,192-record cross-family mappers have different data, sampling and optimization recipes. Their roughly 14.3% request-throughput improvement is a **recipe improvement**, not a causal estimate of doubling training examples. The selected cross checkpoint is epoch 3 of a 24-epoch cosine schedule, not an independently scheduled three-epoch fit. The 16,384-record Llama candidate is effectively tied with the old mapper; retain this negative result.

In `scripts/fit_family_scaling.py`, ZIP uses raw target features with five layer maps and equal layer/context loss; dense uses input RMS normalization and fused-context loss only. Their shared caches make a useful recipe comparison but do not isolate the loss term. ZIP did not win this development search. The 16,384 generated-rollout reproduction simultaneously changes text source, trajectory length, optimization work, objective and execution setting relative to cheap calibration. There is no normalized-dense control on those same generated rollouts.

**Repair:** call these *tested fitting recipes*. State precisely what is held fixed. Describe the full rollout recipe as a successful extension, and keep dense-leading language restricted to the completed capacity/short-text adaptation comparisons. Do not claim that the two-term loss itself caused native parity or that more data alone caused the cross-family gain.

### 5. Token equality is strong finite evidence, not a new lossless-decoding theorem

The new family run contains 128 unique requests per pair, and old/selected maps both match their FP32 AR token arrays on all 128. There are 62,608 Llama tokens and 92,610 cross-family tokens per method; 8 and 24 questions reach the 2,048 cap. These are 128 paired requests per comparison, not 512 independent questions. The two model pairs use the same questions.

The family cohort is the MATH-500 shuffle slice 40:168, excluded from the preceding 40 family-development/check questions. The protocol explicitly disclaims never-exposed status elsewhere in the project. Historical broad evaluation already used MATH-500. Calling this untouched held-out confirmation would mislead reviewers.

**Repair:** say “expanded evaluation with checkpoints and blocks fixed, excluding the recent family tuning questions.” Report direct token-array comparison, stopping/cap behavior and reference precision. No mismatches implies identical deterministic answer scores on these outputs, but does not establish correctness of their mathematics, agreement with BF16 AR, or universal precision independence. Keep the historical BF16 disagreements and scored quality results visible rather than replacing them with the new 100% count.

## Priority 1: positioning, evidence hierarchy and scientific usefulness

### 6. Novelty should be explicit and narrow enough to survive prior work

The current acknowledgment of TriSpec is appropriate. Its primary paper explicitly supports adapter-only tuning with a frozen pretrained EAGLE drafter. Its proxy can approve tokens without full-target verification; this distinguishes its deployment contract from RelaySpec. Frozen adapter learning, cross-target drafter reuse and heterogeneous-vocabulary speculative verification are not firsts. [TriSpec, Section 3.2](https://arxiv.org/html/2601.23180v1); [heterogeneous-vocabulary speculative decoding](https://proceedings.mlr.press/v267/timor25a.html).

**Repair:** the contribution should jointly state the particular source-conditioning replacement, preserved verifier authority, two inherited drafting architectures, source-removal costs, and empirically mapped calibration regimes. Explain why keeping an existing expensive drafter is useful before presenting the simple matrix. Make PARD's zero-new-target-fitting advantage explicit. Treat DFlash/EAGLE-3 as inherited architectures/native controls, not methods RelaySpec broadly defeats. SD2, PARD and TriSpec are different alternatives; an own-runtime AR ratio does not neutralize all runtime and model confounds.

The public-baseline table can stay if its main purpose is practical context. “Highest measured throughput in these public configurations” is supported; “best speculative decoding method” is not. The common-runtime CE/LoRA comparison is stronger evidence for the cheap adaptation decision, but its sixteen-question, 256-token development scope must accompany the claim.

### 7. Explain why the simple map can be useful without a representation-equivalence claim

A useful organizing mechanism is the cost/acceptance tradeoff: removing one source pass saves work per cycle, while mapper error can reduce accepted progress. Better feature regression need not preserve the features decisive for token proposal. The native-interface compression experiments support redundancy of this *consumed interface* in tested settings, not universality of a shared semantic basis.

**Repair:** put one compact, measured mechanism explanation in the main paper. Describe the richer five-layer recipe and its fold into one linear inference projection. The deployment operator remains inexpensive even when calibration becomes richer. Do not equate low parameter count with low entire-pipeline training cost: generating full rollouts and paired capture cost much more than the approximately 14m25 mapper optimization.

### 8. Be candid about remaining evidence gaps instead of generating an unbounded experiment list

Current evidence does not establish multi-seed native parity for the rollout recipe, common-runtime superiority over PARD/SD2, high-concurrency serving behavior, heterogeneous-tokenizer sampling correctness, or broad non-math cross-family quality. These are unrun evidence gaps. Writing can set the scope but cannot close them. No new large GPU campaign is required just to report the completed evidence accurately.

The most useful low-cost analysis is request-level paired intervals for the 128 family comparison and explicit comparison to the old map, keeping fitting/run uncertainty excluded. Separate computational search cost from fitting the selected checkpoint. Main results should retain negative source-reuse cells and the no-gain Llama outcome.

## Recommended allocation of the nine main pages

1. Motivation, frozen-interface deployment contract and three explicit contributions.
2. Method diagram, consumed-interface equations, inference/verification contract and concise positioning.
3. Experimental protocol and historical Qwen source-removal/native-retention results.
4. Quality/precision distinction and workload breadth, with main limitations attached to their claims.
5. Expanded Llama/cross-family table plus richer-rollout native-parity table, visibly distinct protocols.
6. Fixed-work data scaling and frozen small-data confirmation.
7. Capacity/optimization insight and compact native-interface compression result.
8. Matched adaptation and practical competitor differentiation, scoped to measured settings.
9. Remaining synthesis, limitations and conclusion, preserving readable figures and text.

This is a space budget rather than a mandate for nine forced page breaks. To make room, remove the redundant headline speed plot if the main table conveys the same result, combine method illustrations, and move most per-epoch, regularization, layer-choice and precision diagnostic detail to the appendix. The main paper should answer four questions: what can be reused, what does reuse cost/save, how much calibration suffices, and where does the inexpensive recipe stop working.

## Acceptance-oriented assessment

The strongest case is a reproducible deployment contribution with unusually thorough empirical characterization, now including an extension that challenges the initial saturation story. The weakest potential presentation is a catalog of many screens organized around winning numbers, with heavy caveats that arrive only in the appendix. A focused manuscript can make the value clear without promising universal superiority. Remaining novelty and generality concerns should be acknowledged as scientific limits, not hidden by more citations or denser tables.

Round 2 should check that every main claim names the right protocol, that the two-term objective is faithfully specified, that no global “dense always wins” or “shared vocabulary only” statement remains, that the final PDF truly has at most nine readable main pages, and that the expanded result tables regenerate from the raw records.
