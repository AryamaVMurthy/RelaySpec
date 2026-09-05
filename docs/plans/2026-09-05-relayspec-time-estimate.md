# RelaySpec: remaining tasks and time estimate

Updated 5 September 2026. This supersedes the earlier 140–280 active-hour / 350–760 GPU-hour scope estimate. **Reuse all valid existing work. At most four Turing GPUs simultaneously. No experiments were launched to prepare this estimate.**

## Decision-ready estimate

| Quantity | Remaining allowance |
|---|---:|
| Active implementation, analysis and writing | **100–195 hours** |
| New experiments, four GPUs allocated together | **29.25–57.25 elapsed hours** |
| Separate allowance for justified reruns and failed pilots | **12–20 further four-GPU hours** |
| Total compute provision, including that allowance | **165–309 GPU-hours**, approximately **170–310** |
| Calendar target for one lead researcher with coding assistance | **3–5 weeks**; allow **6 weeks** at the upper effort bound and six productive hours/day |
| Aggressive deadline route | **14–20 calendar days**, conditional on the lower effort range, about eight productive hours daily, early correctness resolution and compatible integrations |

These are engineering estimates anchored to completed jobs and a local artifact inventory. They are not measured future runtimes or an acceptance prediction. Scientific validation and raw-artifact recovery remain tasks; this estimate does not mark them complete.

An **active hour** means an hour spent fixing, analyzing or writing. A **GPU-hour** means one GPU allocated for one hour: four GPUs allocated for two hours consume eight GPU-hours. The compute provision therefore corresponds to approximately **41–77 elapsed hours on four GPUs**. It is spread across the project because later runs depend on earlier fixes. Active work and GPU execution overlap; do not add the columns to predict calendar time. Queue waiting is additional and uncalibrated.

## Work already credited

| Existing work | Evidence in the repository | Default action |
|---|---|---|
| Repo cleanup, RelaySpec naming, public GitHub and standardized checks | README, packaging, CI and public remote | Done; only final artifact updates remain |
| ICLR issue review, recent OpenReview study and framing/execution plans | `reports/ICLR_SUBMISSION_REVIEW_2026-09-05.md`, `docs/research/2026-09-05-openreview-review-study.md`, linked plans | Reuse the completed review of 42 public reviews across 11 forums; only targeted updates during writing |
| Four current main comparisons | `reports/final/MAIN_PAIRED_AR.json`, four `*-pairedAR/benchmark-summary.json` files | Recover raw evidence, validate and rescore; no automatic rerun |
| Sixteen historical breadth cells and scored math/code outputs | `reports/FINAL_RESULTS.md`, `reports/final/*-breadth/` | Reuse for their actual source-versus-relay comparisons; fill missing current plain-target comparisons separately |
| Four transfer settings, including a Llama second-family study | `reports/final/TRANSFER_PAIRED_AR.json` | Validate and label the actual intervention; do not schedule a second-family study as if none existed |
| Matched DFlash objective and EAGLE normalization studies | `reports/design-selection/dflash/OBJECTIVE_SELECTION.md`, `reports/design-selection/eagle3/ARCHITECTURE_SELECTION.md` | Reuse valid matched controls; these are not missing experiments |
| Selected connector fits and training measurements | `reports/training/`, `reports/design-selection/` | Reuse weights and summaries when valid; do not repeat fits because the paper wording changes |
| Isolated memory, block-size and cost analyses | `reports/final/EAGLE3_*_MEMORY.json`, `reports/d8-block-ablation/RESULTS.md`, existing cost reports | Reuse measurements under their actual conditions; retain retrospective cost analysis as retrospective |
| Overlap audits, scoring tools, benchmark drivers and paper builders | `reports/final/*OVERLAP_AUDIT.json`, `PROMPT_SIMILARITY_AUDIT.json`, `scripts/` | Complete or repair the existing work |

Artifact existence is established; full scientific validity is not. A change that affects token generation or timing may invalidate an affected measurement while leaving a fitted checkpoint or another workload usable. Existing negative coding results remain part of the evidence.

## Completed main-job time: already spent

Read-only Turing Slurm accounting previously checked on 5 September lists all four jobs as `COMPLETED`, each with four GPUs on node07. Node metadata identifies L40S GPUs; earlier breadth/fitting records include RTX 6000 Ada hardware.

| Completed comparison | Job | Four-GPU duration |
|---|---:|---:|
| DFlash → 8B | 27357 | 1 h 17 m 16 s |
| DFlash → 14B | 27325 | 1 h 51 m 51 s |
| EAGLE-3 → 8B | 27396 | 1 h 49 m 04 s |
| EAGLE-3 → 14B | 27397 | 2 h 44 m 08 s |
| **Already spent, excluded from the remaining base budget** | | **7 h 42 m 19 s** |

If a correction requires replacing all four measurements, the historical runtime suggests about 7.7 four-GPU hours, subject to changed output lengths/runtime. That contingency fits inside the separate reserve above; **do not add it twice**. A whole breadth replacement or substantial algorithm redesign may exceed the reserve and requires a revised estimate. A missing score or incorrect figure label usually calls for offline recomputation, not generation.

## Every required task

The task identifiers match the [execution plan](2026-09-05-relayspec-evidence-execution-plan.md) and completion ledger. GPU durations below mean elapsed time with four GPUs allocated, before the separate reserve.

| Task | Remaining deliverable | Active hours | Four-GPU hours |
| E00 | Recover raw evidence and record reuse decisions | 2–4 | 0 |
| E01 | Resolve target-output differences | 12–28 | 1–3 |
| E02 | Correct equations, data counts and cost statements | 4–7 | 0 |
| E03 | Repair checkpoint export | 2–4 | 0.25–0.75 |
| E04 | Audit data history and freeze final evaluation | 4–8 | 0 |
| E05 | Validate historical main results and add independent final evidence | 3–6 | 7–10 |
| E06 | Add inexpensive adaptation and closest-method controls | 16–32 | 4–8 |
| E07 | Separate distinct training examples from update count | 3–6 | 1.5–3 |
| E08 | Compare linear and nonlinear maps at matched capacity | 5–10 | 2–4 |
| E09 | Connect feature errors to draft behavior | 4–8 | 1–2 |
| E10 | Complete current breadth controls and quality | 4–8 | 6–12 |
| E11 | Validate and correctly label transfer | 3–6 | 0–1 |
| E12 | Measure full setup and representative serving | 12–24 | 4–8 |
| E14 | Check fitting-seed stability | 2–4 | 2–4 |
| E16 | Connect raw evidence to every paper asset | 6–10 | 0 |
| E17 | Rewrite and reconcile the paper | 12–20 | 0 |
| E18 | Final reproduction, PDF and anonymous package | 6–10 | 0.5–1.5 |
| **Total** | | **100–195** | **29.25–57.25** |

The decimal values serve budget arithmetic, not claims of minute-level scheduling precision. Selected warm fitting loops already took approximately 86–119 seconds; data preparation, loading, feature extraction, valid evaluation and interpretation are why an experiment takes longer than its fitting loop.

## Exact reuse assumptions and experiment scope

- **E00:** Reuse existing registries, reports and recovery scripts; verify missing raw artifacts and checkpoint identities.

- **E01:** Reuse mismatch audits and decoder tests; trace first divergence on a small matched-prefix sample before any large run.

- **E02:** Reuse implementation audit, objective-selection studies and training summaries; repair manuscript/accounting offline.

- **E03:** Reuse the existing small reproducer; fix export and verify reload equivalence before drift experiments.

- **E04:** Reuse overlap/similarity audits; inspect flagged cases and distinguish exposed historical data from new final data.

- **E05:** Zero automatic historical main reruns. New allowance is one comparable-size independent final set across four pairs after the data audit; remove it if adequate independent evidence already exists.

- **E06:** Reuse fitting/benchmark infrastructure and valid reference arms; pilot external compatibility, then bounded development screening and selected final comparisons.

- **E07:** Seven configurations share one central point; reuse any matching fits/evaluations. Seed replication is charged only to E14.

- **E08:** Reuse dense and existing nonlinear fits where protocols match; add missing matched-width linear controls, with residual extension conditional on pilot value.

- **E09:** Reuse descriptive geometry; add small common-prefix controls and accepted-prefix/cycle-time probes.

- **E10:** Reuse 16 historical cells and scored outputs. Add only missing current plain-target paired controls/quality evidence. The new domain already charged in E05 is not counted again.

- **E11:** Reuse four completed transfer settings, including the Llama second-family result. Allow a small missing direct-map comparison or quality probe, not a new transfer suite.

- **E12:** Reuse valid isolated-memory measurements and profiling code; add full cold setup and one compatible engine pair at four concurrency settings and three input lengths.

- **E14:** Reuse the original seed when valid; four extra fits across two representative pairs, with evaluation. No best-seed selection.

- **E16:** Repair existing builders/registry bindings rather than rebuilding the pipeline; recompute tables and figures offline.

- **E17:** Reuse the paper, completed related-work review and framing blueprint; write evidence-supported contributions, results and limitations.

- **E18:** Reuse test/build infrastructure; fix stale audit records after actual PDF inspection and run a small clean-environment reproduction.

For E06, the selected plan includes cheap connector/drafter updates plus faithful closest-method comparisons, including SD² and PARD where supported. The budget assumes that small compatibility pilots find usable existing implementations/checkpoints. A major external port is not silently included in a few evaluation hours: if either cannot run faithfully, re-estimate integration and reassess the related claim. E12 similarly assumes a workable engine path after a small pilot; it does not cover building a production runtime from scratch.

E07 starts from seven distinct data/update settings with a shared central point, not eight duplicate settings. Existing matching points are credited. E08 initially compares intermediate widths 256 and 512 with otherwise matched controls; adding a nonlinear branch is conditional on a useful pilot. E14 uses three seeds on two representative families, so valid original checkpoints leave four additional fits. Do not expand every ablation across every model size by default.

A newly timed candidate cannot be paired with an old control from another run and presented as a same-run comparison. Necessary matched controls are part of the affected measurement. Before any launch, record the artifact considered for reuse, the missing question or demonstrated defect, the affected settings and why offline work cannot answer it.

## Optional additions, excluded from the required total

| Addition | Active hours | Four-GPU hours | When it earns inclusion |
|---|---:|---:|---|
| E13: choose the faster deployment method using only earlier calibration data | 6–12 | 2–5 | Required if claiming prediction on unseen requests; otherwise retain the existing measured-cost explanation |
| E15: controlled target-checkpoint drift after export repair | 12–24 | 6–16 | If dynamic adaptation becomes a supported contribution; keep static descendant transfer distinct |
| Actual end-to-end reinforcement-learning extension | Not yet calibrated; allow **3–6+ additional calendar weeks** provisionally | Requires a training pilot | Only after the static paper is sound; static decoding timings cannot price an RL training study |

The first two optional studies add **18–36 active hours and 8–21 four-GPU hours** before their own contingencies. A large transfer-adapter redesign, many extra model families, a new 70B study and unrestricted architecture search are outside this static plan. Adding complexity alone is not a reason to run them.

## Work order and calendar

```text
Recover evidence + correct accounting/export/data history
                         ↓
Resolve decoder correctness and validate reusable measurements
                         ↓
Missing comparisons + controlled ablations + seed checks
                         ↓
Current quality/breadth + transfer checks + serving measurements
                         ↓
Regenerate final evidence → reconcile paper → inspect and reproduce
```

Writing the method/related-work sections and repairing artifact plumbing can proceed while GPU runs execute. Final numerical claims wait for validated evidence. The ledger's completion dependencies do not prohibit preparing these sections early.

| Relative window | Deliverable |
|---|---|
| Days 1–4 | Evidence recovery, factual/export repairs, first-divergence diagnosis and frozen data protocol |
| Days 4–9 | Validate old main results; missing independent final evidence; compatible baseline pilots and seed checks |
| Days 8–14 | Selected final alternatives, data/capacity/mechanism controls, breadth/transfer gaps and serving measurements |
| Days 12–18 | Stable evidence registry, rebuilt figures and complete manuscript revision |
| Days 18–21 | Full claim check, PDF inspection, clean reproduction and anonymous package |
| Additional 1–2 weeks if needed | Upper-range debugging/integration work; scope is reconsidered if a central result changes |

This table is an overlapping target schedule, not a promise that all 195 active hours fit into 21 days. At eight productive hours a day, 100–195 hours is approximately 13–25 workdays. At six hours a day it is approximately 17–33 workdays. Rest days, queue delays and dependency stalls explain the 3–5 week target and upper six-week case.

The [official ICLR 2027 call](https://iclr.cc/Conferences/2027/CallForPapers), checked 5 September, lists abstract registration on **18 September 2026** and the paper on **25 September 2026**, both 23:59 AoE. In India the paper deadline is **26 September at 17:29 IST**. The remaining roughly twenty-day window therefore needs the aggressive route, not the upper-range full estimate. Aim to freeze results around September 18–20 and the PDF by September 23–24.

Do not trade correctness or an essential competing method for cosmetic extensions. If a serving integration cannot be completed, the existing execution plan requires narrowing the serving claim and recording the scope change; the full required-task checker does not automatically become satisfied. Forecasting and drift stay optional.

## Re-estimation triggers

- After the first 12–16 active hours of decoder diagnosis, check whether the cause and a validated repair are known. A continuing unexplained discrepancy changes the completion forecast.
- After one working day per external baseline compatibility pilot, price any real port instead of assuming it is nearly finished.
- After the first representative new evaluation, replace its estimated runtime with measured allocation time on the actual GPU type.
- If raw outputs cannot be recovered, identify exactly which results cannot be regenerated before charging replacement runs.
- If an alternative outperforms RelaySpec, report it and revise the contribution. Method redesign is additional research, not a scheduling buffer.

No more than four GPUs may be allocated across RelaySpec jobs simultaneously. A four-worker job occupies the whole limit. Free resources between runs when work is only analysis or writing. The [machine-readable budget](../../configs/submission/time-budget.json) preserves the previous estimate for history and records the current assumptions and reserve separately. Scientific completion statuses remain unchanged by this planning update.
