# RelaySpec paper framing and evidence blueprint

Date: 5 September 2026. Companion to the [execution plan](2026-09-05-relayspec-evidence-execution-plan.md) and [public-review study](../research/2026-09-05-openreview-review-study.md). This is a writing blueprint, not a revised paper containing completed new results.

## 1. The story to tell

**Recommended title:** *RelaySpec: Retargeting Frozen Speculative Drafters with Linear Interfaces.*

The reader should understand this practical situation: a team already has a trained drafter, then needs to serve a different target model. Its choices include retaining the original source model, obtaining a target-specific drafter, using an independent drafter, or adapting what it already owns. RelaySpec studies the last choice by fitting the input connection while keeping the drafter fixed.

**Central claim to establish:** a correctly fitted interface can make an existing feature-conditioned drafter useful for a new target, while avoiding the original source transformer's decoding cost. The advantage is a measured combination of new-target fitting cost, latency and memory over a stated operating range.

This framing does not need universal semantic alignment, a new general speculative-sampling theorem, or a new reinforcement-learning framework. It does need comparison with inexpensive alternatives. The important distinction from model stitching is the deployment objective and evidence about a frozen speculative drafter's consumed interface, acceptance and cost—not the invention of linear connections between networks.

### Three contributions, contingent on evidence

1. **A retargeting procedure for frozen feature-conditioned drafters.** Explain what is learned and how the DFlash and EAGLE-3 input contracts differ. Evidence: E01/E02/E05.
2. **A measured adaptation and deployment tradeoff.** Compare latency, memory and marginal adaptation cost against source reuse, cheap updates, appropriate target-independent alternatives and native drafters. Evidence: E06/E10/E11/E12/E14.
3. **An explanation of the useful operating range.** Controlled experiments establish the role of interface normalization, fitting data, capacity and accepted progress per decoding round. Evidence: E07–E09. Add prospective selection only if E13 succeeds; otherwise keep it as measured cost analysis.

Use contributions two and three to answer “why should a reader care?” A familiar simple component alone is unlikely to carry the novelty claim.

## 2. A concrete intuition for the first page

```text
Existing reuse:
  original source model → original input features → frozen drafter
                                                       ↓ guesses
                                                   new target checks

RelaySpec deployment:
  new target's available features → fitted interface → same frozen drafter
                                                          ↓ guesses
                                                      new target checks

Fitting only:
  original source features teach the interface what input to supply.
```

The figure must show when target features become available and how the next cycle uses them; it must not suggest the drafter guesses before its conditioning exists. Mark frozen and trainable components. Separate source use during fitting from source absence during deployed decoding. List retained embeddings, output heads, projections and caches in the method table; removing the source transformer does not remove every inherited source weight.

An illustrative explanation, **not a measured RelaySpec result**: source reuse spends 12 milliseconds per round and advances six output tokens, or two milliseconds per token. The relay spends eight milliseconds and advances five tokens, or 1.6 milliseconds per token. Although fewer tokens are accepted per round, it is faster because each round became sufficiently cheaper. Put this example in explanatory material only, clearly labeled; use actual measured values in the paper's main figure.

## 3. Abstract drafting template

> Feature-conditioned speculative drafters offer fast generation, but reusing an existing drafter with a new target requires supplying the input representation it was trained to consume. We introduce RelaySpec, which learns this connection while keeping the target and drafter weights fixed. The interface replaces the original source transformer's role during decoding, with separate input contracts for parallel DFlash and autoregressive EAGLE-3 drafters. Across [VALIDATED TARGETS AND WORKLOADS], RelaySpec achieves [VALIDATED LATENCY RESULT, EXPLICIT BASELINE] using [MEASURED NEW-TARGET FITTING COST] and [MEASURED DEPLOYED MEMORY]. Comparisons with [COMPLETED CHEAP ALTERNATIVES] establish [SUPPORTED COST–PERFORMANCE ADVANTAGE]. Controlled experiments show [VALIDATED MECHANISM AND OPERATING RANGE]. These results make existing frozen drafters reusable under [TESTED DEPLOYMENT CONSTRAINTS].

Do not copy placeholders into an abstract submission. Use the actual numerical range and summary over the declared suite, not only its best cell. If a key proposed experiment is unfinished at registration, write an accurate abstract around completed evidence. The guarantee sentence, if included, must distinguish the standard target-verification argument from verified implementation behavior. Do not write “lossless” merely because older relay/source outputs match.

## 4. Nine-page layout

The page allocations below include figures and tables and total nine pages. They are an initial layout budget, not permission to shrink text below the official format.

| Section | Pages | Reader's question | Required content |
|---|---:|---|---|
| Introduction | 1.0 | What reusable asset and deployment problem does this solve? | Concrete use case, measured source cost, idea diagram, three contributions and closest alternatives. |
| Setting and method | 1.8 | What is trained and what happens during decoding? | Frozen/trainable table; family-specific interface; fitting algorithm; decoding algorithm; verification conditions. |
| Cost and useful operating range | 0.7 | Why can a less accurate drafter still save time? | Measured round-cost decomposition and assumptions; simple break-even explanation. |
| Experimental contract | 0.8 | Are the comparisons valid? | Model/engine/data roles, baselines, output/quality checks, timing units and uncertainty. |
| Main results and adaptation cost | 2.0 | What do we gain, and relative to what? | Main paired table, cost-versus-speed frontier, isolated memory and key quality result. |
| Controlled analysis and transfer | 1.8 | Why does it work and how far does it extend? | Data/update separation, normalization/capacity controls, full task range including regressions, precisely labeled transfer. |
| Related work | 0.6 | What is new relative to the closest methods? | Steering, independent drafting, stitching and drafter training, with relevant distinctions. |
| Scope, limitations and conclusion | 0.3 | When should the method be used? | Concise supported deployment recommendation and observed limits. |

References and appendices follow. Place the required AI-use statement and recommended reproduction statement according to the [current author guide](https://iclr.cc/Conferences/2027/AuthorGuidelines). They do not count toward the main-text limit under the verified September 5 guidance.

### Introduction: paragraph by paragraph

1. State the actual deployment problem in three or four sentences: an existing trained drafter becomes costly to reuse when its source differs from the desired target. Avoid a long generic history of large language models.
2. Explain the available choices fairly. Independent drafters can already serve different targets; target-conditioned drafters exploit target information; steering and adaptation are existing options. The study asks whether keeping a particular inherited drafter frozen is a useful choice.
3. Show the relay intervention and one measured source-cost observation. Explain why the target already exposes useful features and why supplying the old drafter's input is plausible. Do not call compatibility a discovered universal basis.
4. State the exact experimental scope and three contributions. Give one validated cost/performance headline with its comparator. Mention the most important operating boundary in a compact sentence.

### Method: define the quantities before equations

Start with an implementation table containing input layers, concatenated feature width, connector output width, normalization, bias, frozen teacher operation, learned weights, loss reduction, optimizer, data roles and parameter count for each family.

Explain the fitting loss as: “For the same text prefix, make the connector's output resemble the input that the original source supplies to the frozen drafter.” Relative squared error means dividing squared discrepancy by a measure of teacher-vector energy so that large-magnitude vectors do not automatically dominate. State the exact denominator, stabilizer, mask and averaging in code; these details determine the objective.

DFlash's predicted interface includes output normalization. EAGLE-3's selected interface preserves scale. Do not force both into a single normalized formula. If compact shared notation is used, first define a family-specific interface operation and then expand both cases explicitly.

Give two short algorithms. **Fitting:** collect matched-prefix teacher and target features; compute the family-specific connector output; update only its declared parameters; save the interface contract and checkpoint. **Decoding:** obtain current target features; form drafter input; propose tokens; let the target verify; commit the valid prefix plus the appropriate correction/bonus; update caches consistently; repeat until the stopping rule. State where prefill and first-token generation occur.

The correctness argument concerns target-authoritative verification and consistent caches/masks/token boundaries. A connector can make terrible proposals and still leave the ideal output rule correct, at the cost of speed. Finite-precision implementation agreement needs separate evidence. The paper must not claim the feature-matching objective itself guarantees exact output or high acceptance.

### Cost analysis: what the equation answers

The question is the average time paid for each committed output token. Let `C_source` mean total time of a source-reuse decoding round, and `A_source` mean average committed tokens per such round. Define `C_relay` and `A_relay` the same way for RelaySpec. Committed tokens include accepted draft tokens and any target correction/bonus; use the same convention in both arms.

Under comparable output workloads and sufficiently stable round costs, RelaySpec saves decoding time when:

`C_relay / A_relay < C_source / A_source`.

Equivalently, the fraction of accepted progress retained must exceed the fraction of round cost retained:

`A_relay / A_source > C_relay / C_source`.

Each ratio has a direct meaning: retaining 80% of the original progress is useful if the round costs only 60% as much. This is accounting under stated assumptions, not a new general theorem or a prospective prediction. Do not use a mean of cycle ratios when the measured estimator is the ratio of total time to total committed tokens.

The fitting-cost question is different. Let `F` mean the measured **extra** setup seconds of choosing the relay over its comparator, and `D` mean the average seconds saved per request under a fixed workload. If `F` is positive and `D` is positive, at least `ceil(F / D)` requests are needed to repay that extra setup. `ceil` means round upward to a whole request. If savings are zero or negative, there is no finite repayment point. If setup is already cheaper, report that directly. This calculation needs uncertainty and a stated workload; it is not a workload-independent promise.

## 5. Figures and tables with exact evidence requirements

| Artifact | Purpose and axes/columns | Evidence | Placement |
|---|---|---|---|
| Figure 1: method and source cost | Fitting/deployment paths; frozen/trainable components; measured cost split | E02/E12 | Introduction |
| Table 1: main paired performance | Target, drafter, method, request seconds, target tokens/s, speed versus plain target, speed versus source reuse, current quality/agreement | E01/E05/E14/E16 | Main |
| Figure 2: adaptation versus deployment | Horizontal: measured new-target GPU-seconds; vertical: steady-state tokens/s; panels for drafter/target; marker labels for memory and method | E06/E12 | Main |
| Table 2: resource accounting | Distinct fitting records, token presentations, trainable/retained parameters, warm/cold time, deployed peak memory | E02/E06/E12 | Main |
| Figure 3: mechanism | Round time and accepted progress; data versus updates in separate panels; main interface ablation | E07/E09 | Main; complete sweeps appendix |
| Figure 4: workload range | All planned task cells, same explicit baseline, uncertainty and reference line at 1 | E10 | Main |
| Table 3: transfer | Source drafter, target, unchanged/refitted map, new weights, tokenizer conversion, cost, quality and speed | E11 | Main if central; detailed protocols appendix |
| Serving plot | Concurrency versus throughput and request latency, named engine; source-residency memory panel | E12 | Main if deployment is a contribution; full settings appendix |
| Forecast plot | Calibration-frozen predicted choice versus observed held-out outcome, including wrong choices and overhead | E13 | Only if completed; otherwise omit forecasting claim |

The layout may combine figures to fit nine pages, but not compress away the cost, quality or negative cells that determine the conclusion. Every plotted coordinate, label, interval and caption must come from the same run registry. No manually refreshed annotations on old points.

Do not compare totals across different tokenizers as equivalent generated-token workloads without defining the counting convention. Do not label request-time speedup as throughput when lengths differ. Use absolute timings alongside relative improvements.

## 6. What goes in the main paper, appendix, or quarantine

Choose placement by relevance and evidential validity before observing a new result's sign.

| Information | Decision | Why |
|---|---|---|
| Core comparisons, current quality/equality and cheap adaptation | Main paper | These determine whether the contribution exists. |
| Known code regressions and the scope they imply | Main summary plus complete appendix matrix | They determine where reuse is worthwhile. |
| All seeds, complete workload cells, trace categories, configs and hardware | Appendix and artifact | Necessary detail; main text retains central uncertainty and interpretation. |
| Full tensor dimensions, loss reduction, fitting text construction and exact cost scope | Short main table plus complete appendix | Readers must understand what was actually optimized. |
| 524M transfer adapter | Main cost if its result is highlighted; otherwise full labeled appendix experiment | Calling it small would misrepresent its role. |
| Old results with corrupted exports or missing essential provenance | Quarantine in repository audit, excluded from scientific result tables until repaired | These are invalid/incomplete evidence, not unfavorable evidence to hide. |
| Older superseded configurations | Repository history and an experiment-selection log | Preserve provenance without making them a second competing result set. |
| Common-teacher similarity pictures | Appendix as descriptive probes, or omit if they add no insight | They do not establish shared semantics. |
| Online RL/drift story without a valid complete comparison | Future work, or omit from this paper | It is a different empirical claim. |
| Repeated background, duplicate figures and unsupported superlatives | Remove | They consume space without supporting the argument. |

An unfavorable but valid result belongs in the declared complete study. An invalid run belongs in an audit with the reason for invalidity. This distinction prevents both misleading cherry-picking and cluttering the paper with broken experiments.

## 7. Positive, precise wording

These replacements express the strongest claim the evidence could support. Fill in measurements only after validation.

| Avoid | Prefer |
|---|---|
| “All prior methods retrain a drafter for each target.” | “RelaySpec reuses an existing feature-conditioned drafter by adapting its input interface; we compare this choice with target-independent drafting and inexpensive adaptation.” |
| “A universal shared basis enables arbitrary transfer.” | “Across the tested targets, a fitted linear interface supplies features that the frozen drafter can use effectively.” |
| “Training-free” | “The target and drafter remain frozen; only the interface is fitted.” |
| “Trained in under two minutes” without scope | “The measured fitting loop took [X] seconds on [hardware]; complete setup took [Y].” |
| “8,192 training examples” for two passes | “4,096 distinct records, presented twice.” |
| “Unlabeled prompts” for problem-plus-solution text | “Feature supervision on [exact text construction], without a direct answer-token training loss.” |
| “Exact prediction of every speedup” from the same runs | “Measured cycle costs explain the observed speed tradeoff.” |
| “Nonlinearity is worse” from unequal capacities | “Under the matched configurations evaluated, [specific comparison and interval].” |
| “Works across all tasks” | “The relay is beneficial in [validated regimes]; the complete task matrix identifies [observed boundary].” |
| “Zero-shot transfer” after fitting a new adapter | “The drafter is reused, with [specified new adapter/map] fitted for the new target.” |
| “Nearly identical quality” from old source-reuse tests | “On the current paired outputs, target accuracy differed by [difference and interval]; exact output agreement was [count].” |

A concise boundary sentence can strengthen the main result: “The savings are largest when the removed source computation outweighs the reduction in accepted progress; the coding results identify cases where that condition does not hold.” Use this only if the measured decomposition supports it. A result can remain faster than plain target decoding while being slower than source reuse—name the comparator.

## 8. Decide the final story from results, not desired numbers

| Validated outcome | Appropriate conclusion |
|---|---|
| Relay beats source reuse, approaches native speed, and costs less to adapt than cheap updates | Strong main story: a useful retargeting tradeoff. Report the complete range, not only the maximum. |
| Native target drafter is faster | Compatible with the story if RelaySpec demonstrates a meaningful adaptation or memory advantage; show both. |
| A cheap update is uniformly faster and cheaper | The present novelty claim weakens. Analyze a real frozen-weight/deployment constraint or revise the method; do not remove the baseline. |
| Linear and nonlinear are statistically indistinguishable | Prefer the simpler implementation when costs justify it; do not claim linearity is intrinsically superior. |
| Extra unique data helps but extra updates do not | Support a data-diversity explanation within the tested range. |
| Extra updates help but more distinct data does not | Support an optimization explanation within the tested range; correct the scaling story. |
| Cross-family or cross-tokenizer fails | Narrow the generality claim and report the tested boundary. Same-family usefulness can still be real. |
| Engine gains shrink with concurrency | State a latency-oriented operating range and show the throughput tradeoff. |
| Decoder mismatch remains unexplained | Stop the lossless/drop-in claim; more speed experiments do not resolve it. |
| Forecast does not beat a fixed choice after calibration cost | Retain cost accounting, remove prospective-selection contribution. |

## 9. Direct changes to the current manuscript

In `paper/iclr2027/relayspec_iclr2027.tex`:

- **Title/abstract/introduction:** use the shorter title and concrete reuse setting; replace cost/data multipliers and unsupported uniqueness; emphasize marginal adaptation cost.
- **What one decoding cycle does:** retain its helpful intuition, add the whole-system diagram and exact first-token/cache/stopping behavior from E01.
- **Method:** replace the unified normalized account with family-specific contracts; correct optimizer and loss reductions; distinguish a regression surrogate from the actual DFlash objective.
- **When is this worth doing?:** retain the accepted-progress/cost explanation, label current reconstruction retrospective; include a forecast only after E13.
- **Setup:** expose fitting solution text, data-history roles, distinct records/presentations, warm versus complete setup time, hardware and method order.
- **Does it work, and what does it cost?:** replace mixed-run assets with E16 output; include current target-quality/equality, native and cheap-adaptation controls, absolute times and memory.
- **How far can one frozen proposer be moved?:** separate unchanged map, new adapter, newly fitted second-family map and tokenizer conversion; replace semantic-basis wording with measured compatibility.
- **Related work:** correct PARD and SD² classifications; distinguish the contribution from stitching, target-independent drafting, drafter training and online adaptation.
- **Limitations/conclusion:** summarize the observed operating range in plain positive language; retain the material restrictions on model access, hardware, tasks, precision and tokenizer support.
- **Appendices:** update the solver argument, parameter counts, full task/seed tables, overlap measure, data construction, memory protocol and cross-family procedure. No obsolete statements may contradict the main text.
- **AI-use/reproduction statements:** accurately describe the actual assistance and artifact; use an anonymous supplementary snapshot during review.

## 10. Final reader test

Before submission, ask a technically competent reader to explain, without the appendix: what was reused; what was trained; what the target verifies; which baseline each speed compares with; where fitting costs enter; why acceptance can fall while speed rises; which observed settings do not benefit; and what distinguishes the work from the closest cheap alternative. If any answer requires guessing, revise the paper rather than adding more terminology.
