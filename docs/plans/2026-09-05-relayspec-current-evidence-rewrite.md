# RelaySpec manuscript rewrite from current evidence

Date: 5 September 2026. This is the executed writing plan and evidence-selection record for the current manuscript. It narrows the present paper to recorded source-reuse comparisons. The broader research execution plan remains active. No GPU experiments were launched for this rewrite.

## Paper question and contribution

An existing feature-conditioned drafter expects states from its original source model. The current paper asks whether a learned map from a new target's states can replace that source transformer's computation while keeping the drafter fixed. It reports the incremental throughput and memory change relative to optimized source reuse under a shared verifier, together with task outcomes.

The empirical unit is a paired drafter/target/workload comparison. The core uses two released drafter implementations and two Qwen3 target sizes. Source reuse is a precisely defined reference for the intervention, not a claim to cover every deployment alternative. The prose starts with the procedure, explains the cost balance, and then gives measured results.

## Evidence selected for this version

| Evidence | Use in the paper | Validation performed during rewrite |
|---|---|---|
| Four recorded source-reuse MATH-500 runs | Main throughput, score, agreement and output length | Raw rank files checked for complete unique pairs. Throughput and token agreement recomputed. Saved math scores matched to raw completions and summary accuracy. |
| Sixteen fixed-map workload cells | Complete plot and appendix table, including all directions | Raw timings, token counts and agreement checked against the matrix. Complete two-turn runs used for EAGLE MT-Bench. |
| Recorded math and code quality | Main quality table | Math/GSM8K scored rows checked against raw output identity. EvalPlus task outcomes and pass counts checked against saved scorer result files and summary values. This is a consistency check of recorded scoring, not a fresh execution of external scorers. |
| 468-question development complement | Separate appendix analysis | Existing subset summaries retained and hashed. It is identified as a subset of saved runs, not newly untouched evaluation. |
| Four selected fitting summaries | Per-map fitting-loop time | Existing loop timings reused, with the timer boundary stated. |
| Separate EAGLE memory processes | Main memory figure | Existing isolated-memory artifacts reused. Eight paired prompts per target stated. |
| Matched EAGLE normalization and DFlash objective comparisons | Design explanation and appendix tables | Tables generated from recorded analysis/selection artifacts. EAGLE loss values read from the recorded architecture-selection report. |
| Prompt overlap and similarity audits | Main data qualification and complete appendix | Existing exact and token-set overlap artifacts reused, with the correct metric definition. |

The generated registry is `paper/iclr2027/generated/evidence_registry.json`. It records input hashes and consistency checks. The figure/table generator fails if raw pairs are incomplete, duplicated, or inconsistent with selected reported results. It also checks per-task code outcome counts. These checks establish artifact consistency, not target-decoder equivalence.

## Material corrections

- The DFlash loss compares output-normalized vectors. The selected EAGLE map preserves raw input scale. The paper now gives both interfaces explicitly.
- Fitting inputs contain problem and solution text. They are not called unlabeled prompts. The objective uses source-derived features rather than answer correctness labels.
- Fitting-loop time is separated from model loading, checkpoint writing and full deployment setup.
- Removing the source transformer is distinguished from removing source embeddings or vocabulary projections used by a drafter.
- Task score agreement and token-sequence agreement are separately defined and always name source reuse as their reference.
- The cost equation is approximate and uses same-run measured quantities. It is framed as accounting, not a demonstrated prospective prediction rule.
- The complete breadth matrix is retained. A ratio below one remains visible. The explanation describes the balance of source cost saved and extra cycles.
- PARD's target independence and SD²'s frozen-drafter variant are credited. The exclusive frozen-drafter novelty claim and cross-paper cost ranking are removed.
- Model stitching is credited. The paper makes behavioral claims rather than claiming a shared semantic basis.
- The runtime versions and random seeds appear in the appendix. The main prose focuses on the experiment and its interpretation.
- The AI statement describes actual assistance without claiming that human authors have already inspected every artifact or approved submission.

## Evidence outside the current manuscript boundary

The following artifacts remain in repository history/evidence directories. Their omission is based on claim support and measurement completeness, not on whether the result is favorable.

| Artifact or claim | Reason for keeping it outside this version | Condition for promotion |
|---|---|---|
| New plain-target paired headline ratios | Raw local provenance and current-output quality/correctness questions are unresolved. The old draft mixed these ratios with older graph coordinates. | E00/E01/E05 establish the exact run, task quality, verifier behavior and valid comparison. |
| Four exploratory transfer settings | Different interventions were conflated, including a 524M adapter and tokenizer conversion. Their current quality/verification evidence needs validation. | E11 supplies precisely labeled interventions, paired quality and valid verification, with E01 first. |
| Claim that a dense adapter is small or economical | Its 524M parameters exceed a direct 65.5M map. | Report actual costs and a fair direct-map comparison. |
| Closed-form and compressed nonlinear superiority claims | Different objectives and capacities prevent the conclusions previously drawn. | E07/E08 supply matched questions and controls. |
| An 8,192-distinct-example scaling claim | The run repeats a 4,096-record manifest and changes the update count. | Report presentations accurately or complete independent data/update sweeps. |
| Drift/export conclusions | The exporter has a demonstrated reload discrepancy. | E03 validates export before E15 conclusions. |
| Prospective no-slowdown policy | Same-run accounting does not establish prediction on unseen data. | E13 freezes calibration and tests the rule on unseen requests with overhead charged. |
| Cross-paper training-cost superiority | Initial drafter creation and marginal retargeting cost are different quantities, and published hardware/budgets differ. | E06/E12 provide matched adaptation comparisons and complete cost boundaries. |

The related conditions and narrower empirical scope remain visible in the manuscript. The rewrite does not establish a general equivalence to plain target decoding by dropping that baseline's unresolved measurements.

## What was learned from accepted papers and public reviews

The existing study covers 42 public reviews across 11 discussions, with nine accepted papers. Reuse that work rather than repeating the survey. This turn rechecked the ICLR author guidelines and primary paper pages/full text for PARD, DFlash, EAGLE-3, SD², model stitching and the ICML 2025 interpretation study. Attempts to retrieve several OpenReview notes again returned a browser challenge or HTTP 403. Therefore this rewrite uses the previously recorded review study and its exact note links, without claiming a fresh read of inaccessible notes.

| Source of editorial guidance | Applied change |
|---|---|
| PARD, ICLR 2026, and its baseline-fairness discussion | Define the target dependence being removed and hold the proposal policy fixed within the central comparison. Credit target-independent drafters. |
| RepSpec, ICLR 2026 | Give existing linear structure a concrete purpose. Avoid an architectural claim from an unmatched nonlinear comparison. |
| EAGLE-3, NeurIPS 2025 | State exact feature layers, fitting records/updates and implementation details. Separate interface choices from the main evaluation. |
| SWIFT, ICLR 2025 | State precisely which adaptation costs are timed. |
| Faster Cascades, ICLR 2025 | Explain the objective and intuition before equations. |
| OmniDraft, NeurIPS 2025 | Separate measured deployment behavior from projections and distinguish changed weights from reused weights. |
| Heterogeneous-vocabulary work, ICML 2025 | Keep target verification and output comparability explicit. |

These are lessons about clear, fair reporting, not an acceptance formula. See `docs/research/2026-09-05-openreview-review-study.md` for primary links and the distinction between reviews and editorial inference.

## Execution completed and next experiments

The rewrite replaces the title, abstract, main sections, method equations, related work, conclusion and appendices. It rebuilds six vector figures and all current result tables from a consistent source-reuse evidence boundary. Obsolete rendered assets are removed from the active paper directory while raw experiments and scripts remain available.

The next scientific work stays in the existing execution plan, in this order:

1. E00/E01: recover exact-run artifacts and resolve target-output differences with a small controlled diagnostic. Preserve unaffected measurements and checkpoints.
2. E06: compare the relay with inexpensive drafter updates and faithful supported closest methods under matched costs. This most directly tests whether the demonstrated operation is a compelling choice.
3. E04/E05/E10: freeze exposure history and add only necessary independent evaluation/current quality evidence. Retain all valid historical results.
4. E07/E08/E14: separate distinct examples from update count, match linear/nonlinear capacity, and add the missing representative seeds.
5. E12: measure full setup and one supported serving setting under concurrency. Reuse existing isolated-memory experiments.
6. E11: promote validated transfer settings with their exact trained/reused weights and costs. E13 forecasting and E15 drift remain optional.

The rewrite changes paper claims, not the research ledger's evidence requirements. Formatting or a passing artifact check does not complete the 17 required scientific tasks. The separate time budget and four-GPU cap remain in force.

## Final artifact verification

The revised manuscript has nine main pages and 15 total pages, six vector figures,
nine tables and 20 cited sources. Ruff lint and formatting checks pass. All 203
tests pass, including the built-manuscript checks. The manuscript audit passes
for official style hashes, anonymity, page count, references, generated evidence,
embedded fonts, PDF parsing and the exact-hash visual-review record. Every page
was inspected in color and grayscale. No new GPU experiments were run for this
rewrite. The saved paired task outcomes cover 4,680 math/code problems and agree
between the two compared methods under the recorded scorers.
