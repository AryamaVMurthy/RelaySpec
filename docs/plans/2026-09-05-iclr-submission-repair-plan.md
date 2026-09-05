# RelaySpec ICLR 2027 repair and improvement plan

**Expanded plan:** use the [detailed execution plan](2026-09-05-relayspec-evidence-execution-plan.md), [paper framing blueprint](2026-09-05-relayspec-paper-framing-plan.md), and [public-review study](../research/2026-09-05-openreview-review-study.md) for current planning. This initial overview is retained for context.

Date: 5 September 2026. This plan follows the [submission review](../../reports/ICLR_SUBMISSION_REVIEW_2026-09-05.md). It proposes work; it does not claim the experiments below have run. No GPU jobs were launched as part of the review.

**Objective:** submit a trustworthy paper about low-cost retargeting of existing frozen speculative drafters. Correctness and fair comparisons come before additional architectural complexity. The date windows are planning targets, not estimates of measured run time.

## Phase 1 — Establish which evidence can be trusted: 5–7 September

| Priority | Task | Concrete deliverable | Completion condition |
|---|---|---|---|
| First | Retrieve latest paired-AR and transfer artifacts. | Raw per-request outputs/times, scored answers, configs, source snapshots, checkpoint revisions, data hashes, and environment versions for every headline row. | Another person can identify exactly which files produced each main number. |
| First | Diagnose low agreement with native autoregressive decoding. | First-divergence report covering token IDs, decoded text, stopping behavior, cache/mask/position parity, precision, and matched-prefix target scores. | All observed mismatch categories are accounted for; no unsupported “all numerical” or “tokenizer caveat” explanation remains. |
| First | Validate drift checkpoint export. | Zero-training adapter export/reload comparison; actual remote checkpoint key audit; target quality checks before and after export. | Plain loading has no unintended missing or unexpected weights and preserves outputs within the defined precision check. Only then interpret training drift. |
| First | Correct method description. | DFlash and EAGLE-3 equations traced to actual consumed tensors and loss code. | Normalization, teacher interface, loss weighting, and optimizer settings agree with implementation. |
| First | Correct immediate factual claims. | Reviewed edits for 4,096 distinct records versus 8,192 presentations, 524M adapter, nonlinear capacity, source-trunk removal, and warm fitting cost. | Every numerical count is computed from the actual config or artifact. |
| First | Rebuild novelty taxonomy. | SD² frozen variant, PARD target-independent setting, stitching precedent, and marginal-cost columns. | Every “only,” “first,” and “per-target training” statement has directly supporting evidence or is removed. |

**Decision at the end of this phase:** if native-decoder correctness remains unresolved, continue diagnosis and narrow the claims. Do not paper over it with older source-reuse agreement. If drift artifacts fail validation, quarantine those results and postpone the drift extension; the core static-retargeting paper can proceed independently.

## Phase 2 — Re-establish the core experiment: 8–12 September

Create a single frozen evaluation protocol before new comparisons. Record which existing examples were used for development. A subset inspected during architecture or hyperparameter selection cannot later be called untouched.

Run the existing main model pairs with identical prompts, stopping rules, precision, target implementation, and measurement definitions. Include plain target decoding, optimized source reuse, the relay, and available target-specific drafters. Rotate or randomize method order to reduce systematic warmup or thermal effects. Use representative warmup and avoid overlapping competing jobs on the timed GPU.

For each arm save request latency, generated tokens, total tokens per second, accepted prefix lengths with the counting convention, task correctness, and token/text agreement against the declared reference. Report prefill and decoding separately where measured. Do not combine an older quality result with a new speed run as though they were one experiment.

Repeat at least three relay fits on representative DFlash and EAGLE-3 pairs if resources allow. The purpose is to determine whether the result depends on a lucky fit. Expand to all pairs only if meaningful instability appears or resources permit; avoid spending the whole window repeating a stable result.

Repair the ablation design:

1. At fixed distinct training records, vary update count. This asks whether extra practice helps.
2. At fixed update count, vary distinct training records. This asks whether more varied examples help.
3. Compare the actual selected interface with its normalization alternatives, separately for each drafter family.
4. Compare the linear map with an equally compressed two-matrix linear map and a comparable nonlinear map. This separates compression from nonlinearity.
5. Treat the current ridge fit as a different objective unless the full loss, regularization, and example weighting are genuinely matched.

**Completion condition:** the main speed/quality table is regenerated from raw evidence; any remaining implementation-level mismatch is accurately characterized; the central method and controlled claims are supported without relying on the known confounds.

## Phase 3 — Add the experiments most likely to change a review: 13–17 September

Execute in this order, stopping lower-priority expansion if the core work overruns:

| Order | Question | Experiment | Why it is worth the budget |
|---|---|---|---|
| 1 | Is a relay the best cheap way to adapt an already trained drafter? | Same fitting records and measured compute budget for RelaySpec and a suitable small drafter update/selective update or faithful frozen-steering baseline. Include total new parameters and source residency. | Directly tests the contribution against inexpensive alternatives rather than expensive from-scratch training. |
| 2 | Does the result hold outside the fitted math distribution? | Freeze the map and test a previously unexposed domain; complete native-AR breadth where practical and preserve known code regressions. | Supports scope and failure boundaries. |
| 3 | Can the cost rule make a decision in advance? | Measure on calibration requests, freeze the rule, then forecast relay-versus-source decisions on separate requests/tasks. | Converts an explanation based on observed timings into an actual prediction claim. |
| 4 | Is this tied to Qwen and this execution implementation? | Validate the second-family result with quality evidence; if time permits, one optimized-serving-engine experiment at stated concurrency. | Tests portability more directly than several additional same-engine Qwen rows. |

The existing adapter result can remain as a clearly labeled transfer experiment with its full 524M cost. Do not prioritize making it look cheaper unless parameter-efficient adapter reuse becomes part of the paper's actual central claim.

By abstract registration, use only statements already justified by completed evidence. Do not register an abstract that depends on an unvalidated drift experiment succeeding.

## Phase 4 — Freeze results and rewrite: 18–22 September

Build all figures and tables from one raw-data pipeline. Use stable experiment IDs and record exact model and source revisions. Check that graph coordinates and numerical annotations come from the same runs and metric.

Rewrite around these questions, in order:

1. What dependency prevents reuse of an existing drafter?
2. What exact input does each drafter consume, and what is learned to replace it?
3. Does target verification and the implementation preserve the claimed behavior?
4. What speed, memory, and fitting cost result under fair comparisons?
5. Why does it help on some pairs and fail on others?
6. How much evidence supports transfer to a new family or changed target?

Keep correctness, the adaptation-cost comparison, and important regressions in the main paper. Put exhaustive configurations, all seeds, trace categories, and additional model pairs in the appendix. Make every table baseline explicit. Separate limitations demonstrated by experiments from possible future limitations.

Prepare a reproduction entry point that builds tables from archived small artifacts without needing access to large-model weights. Separately document how to reproduce expensive runs, including expected hardware and approximate measured duration. Refresh README, claim-evidence mapping, and completion status to the same frozen result set.

## Phase 5 — Final submission audit: 23–24 September, with buffer before the 25 September AoE deadline

- Build the final PDF using official ICLR 2027 style files. Check the nine-page main-text limit, anonymous content/metadata, bibliography, appendix boundaries, and required AI-use statement against the current official guidance.
- Inspect every page visually at readable size. Repair title wrapping, small chart labels, overfull content, and captions that change the comparison baseline.
- Update the visual-review record only for the actual inspected final PDF hash and page count.
- Run the required repository checks once after the final changes. Resolve real failures; do not weaken tests merely to get a green result.
- Recompute each headline number from its archived run and verify units, denominators, sample counts, confidence intervals, and referenced experiment IDs.
- Verify the supplementary package from a clean environment and ensure that the public/reviewer artifact does not require private cluster paths to understand results.
- Complete abstract registration and full submission before the official deadlines rather than using the time-zone conversion as the working target.

Official dates: [abstract 18 September and paper 25 September 2026, both 23:59 AoE](https://iclr.cc/Conferences/2027/CallForPapers). In India these correspond to 19 September and 26 September at 17:29 IST. Check [author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) during the final audit.

## What to postpone

Postpone a new reinforcement-learning rollout stack, elaborate drift controllers, many more cross-tokenizer pairs, and broad claims about a universal shared hidden-state basis. They introduce new failure modes and close prior work before the current evidence is stable.

After the submission is sound, the most interesting longer-term direction is to learn from the target's own verification feedback while preserving a fixed drafter. Begin with a validated target checkpoint sequence, measure how the source-reuse control changes too, and compare directly with [SpecRoll](https://arxiv.org/abs/2608.04962), [OnlineSPEC](https://arxiv.org/abs/2603.12617), and [FastGRPO](https://openreview.net/pdf?id=zuGt6TYYtS). The question is whether a cheap input correction is sufficient, when it must be updated, and when the drafter itself needs adaptation.

## Minimum defensible submission

A coherent submission can be smaller than the current collection of claims. The minimum strong version has validated target behavior, correctly specified interfaces, reconstructible paired results, an honest marginal-cost comparison with a cheap alternative, one controlled explanation of the tradeoff, and clearly stated failure boundaries. Additional transfer breadth is valuable only after these are secure.
