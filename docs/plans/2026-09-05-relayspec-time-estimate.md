# RelaySpec time and compute estimate

Prepared 5 September 2026 after inspecting repository timings and live, read-only Slurm accounting on Turing. **Resource limit: four GPUs simultaneously**, as authorized by the user. No jobs were launched for this estimate.

**Reuse amendment, 5 September:** all valid existing work must be reused. The four main jobs below are completed, and their 7 h 42 m runtime is already spent. The estimates retained below describe a conservative expanded scope with replacement contingencies; they are **not an audited list of remaining runs or a commitment to repeat completed experiments**. A remaining-work total is pending the artifact reuse inventory in E00. Documentation fixes, offline rescoring and figure regeneration do not by themselves require new generation. New launches must identify a specific evidence gap or affected defect as described in the [execution plan](2026-09-05-relayspec-evidence-execution-plan.md#2-work-order-and-result-gates).

## Recommendation

- **Focused submission:** approximately **3–4 calendar weeks** for one lead researcher working with coding assistance, assuming six productive hours per day and six working days per week. An aggressive **14–20 calendar days** is possible with roughly eight focused hours daily, prompt GPU access, early correctness resolution and reduced optional scope.
- **Full strong static-retargeting plan:** approximately **4–8 calendar weeks**, with **140–280 researcher-hours** and a provision of approximately **350–760 GPU-hours**, including a reserve. This includes the cheap alternatives, controlled analyses, second-family replication, serving integration and complete paper/artifact work.
- **Everything plus a substantive online reinforcement-learning extension:** provisionally **6–12+ weeks**. This is a lower-confidence research estimate requiring a separate pilot and final model-quality evaluation; it is not just another static benchmark.

These are effort estimates, not an acceptance forecast. The earlier September execution windows were aggressive scheduling targets. They should not be read as evidence that the entire expanded plan fits comfortably into twenty days.

## Measured starting point

The latest main jobs were run on Turing node07. Current Slurm metadata labels this node `l40s`; earlier fitting and breadth records include RTX 6000 Ada hardware. Therefore the latest main-job measurements below are the better baseline for scheduling equivalent work on four L40S GPUs. Other GPU types require a timing pilot before extrapolation.

| Main MATH-500 job | Methods | Actual four-GPU allocation duration |
|---|---:|---:|
| DFlash → 8B, job 27357 | 4 | 1 h 17 m 16 s |
| DFlash → 14B, job 27325 | 3 | 1 h 51 m 51 s |
| EAGLE-3 → 8B, job 27396 | 4 | 1 h 49 m 04 s |
| EAGLE-3 → 14B, job 27397 | 4 | 2 h 44 m 08 s |
| **Total, sequential on the same four GPUs** | | **7 h 42 m 19 s** |

Source command: `sacct -j 27357,27325,27396,27397 --allocations --parsable2 --format=JobID,Submit,Start,End,Elapsed,AllocTRES`. All four allocation records reported `COMPLETED` with `gres/gpu=4`. These durations establish resource use, not scientific validity of the results.

The four local benchmark summaries contain 28.74 summed request-hours, which gives an ideal balanced four-worker estimate of 7.18 hours. The actual allocation total is 7.71 hours. The difference is why this estimate uses job accounting rather than simply dividing summed request time by four.

The selected fitting summaries report about 86–119 seconds for their fitting loops. These exclude model-loading time and do not imply that a complete new experiment takes two minutes. Preparing a valid configuration, debugging it, evaluating outputs and interpreting evidence take much longer than the fit itself.

One GPU-hour means one GPU reserved for one hour. Four GPUs reserved for eight hours consume 32 GPU-hours. Their memories do not automatically combine into one model's usable memory. The current benchmark generally distributes requests across workers; it does not make an individual request four times faster.

## Full static-plan breakdown

Researcher-hours mean active implementation, diagnosis, checking and writing effort. Four-GPU hours mean elapsed allocation time while four GPUs are reserved together, excluding queue wait. The two columns overlap: a researcher can write or analyze earlier results while a benchmark runs. Do not add them together to predict calendar duration.

| Work | Tasks | Researcher-hours | Four-GPU hours |
|---|---|---:|---:|
| Raw evidence, equations/accounting and data protocol | E00/E02/E04 | 15–30 | 0.5–2 |
| Decoder diagnosis and checkpoint-export repairs | E01/E03 | 19–46 | 4.5–14 |
| Main comparisons and representative fitting seeds | E05/E14 | 6–12 | 19–30 |
| Inexpensive adaptation baselines | E06 | 16–32 | 8–20 |
| Data/update, capacity and mechanism analyses | E07/E08/E09 | 20–40 | 9–22 |
| Task breadth and transfer validation | E10/E11 | 14–28 | 14–30 |
| Serving integration, full setup and memory | E12 | 16–32 | 8–20 |
| Unified evidence-to-paper pipeline | E16 | 10–18 | 0 |
| Paper rewrite, figures and evidence interpretation | E17 | 16–28 | 0 |
| Final reproduction and submission inspection | E18 | 8–14 | 2–4 |
| **Total before compute reserve** | | **140–280** | **65–142** |

The GPU estimate is **260–568 GPU-hours before reserve**. To leave one-quarter of the total allocation available for reruns, divide those figures by 0.75: approximately **347–757 GPU-hours**, rounded to **350–760** for planning. This corresponds to roughly **87–189 elapsed hours on four GPUs**, or **3.6–7.9 days of continuously allocated compute**. That compute is spread over several weeks because most runs depend on implementation and analysis completed earlier.

The researcher ranges already include ordinary iteration. A fundamental algorithm redesign, inaccessible baseline, major environment migration or unavailable data could exceed them. These future-task ranges are engineering judgments anchored to measured jobs; they are not measured durations or calibrated probability intervals. [Machine-readable budget](../../configs/submission/time-budget.json).

### How the compute ranges were constructed

- **Main evaluation, E05:** the earlier 16–24 four-GPU-hour allowance assumed replacement of the historical four-pair suite, a comparably sized new final evaluation and modest reruns. This is a contingency, not mandatory remaining work. Reuse of all valid historical main evidence removes the approximately 7.7-hour replacement from that allowance. If only some configurations are affected, charge only those reruns and needed matched controls. Additional independent evaluation remains a separate question. Do not subtract the same saved hours again from other tasks or assume reuse is validated before the audit.
- **Seed stability, E14:** budget 3–6 hours for additional connector fits and candidate evaluation on representative pairs. Reuse identical fixed reference outputs only when prompt, model, software, precision and measurement contracts match; otherwise re-run controls. Do not select the best seed.
- **Cheap baselines, E06:** budget two representative drafter families, three fitting budgets, development screening and final evaluations of the selected alternatives. Existing source/native controls can be reused only under an identical validated protocol. External integration effort is the greater uncertainty.
- **Ablations, E07–E09:** screen on development subsets, then run frozen selected comparisons on final data. The estimate covers the stated small sweeps, not unrestricted hyperparameter search.
- **Breadth/transfer, E10/E11:** add current plain-target and available native controls, a new workload and selected transfer cases. Historical short two-arm breadth runs are not adequate estimates for a new suite with slow plain-target arms.
- **Serving, E12:** includes a compatibility pilot, one representative pair, four concurrency settings, input-length strata and isolated memory measurements. It excludes a full production service, large-scale distributed serving, and a new 70B study.
- **GPU reserve:** covers failed pilots, remeasured controls and repeat measurements. It does not turn blocked implementation time into runnable work.

## What fits before the current deadline

The official [ICLR 2027 call](https://iclr.cc/Conferences/2027/CallForPapers) lists abstract registration on September 18 and the full paper on September 25, 2026, at 23:59 AoE. In India, the paper deadline is September 26 at 17:29 IST. From September 5, this is approximately twenty days. Aim to freeze the paper by September 23–24.

For this window, choose a focused version requiring roughly **90–130 researcher-hours** and provisionally **180–320 GPU-hours including rerun allowance**. This is a reduced scope, not the whole 140–280-hour plan. It includes:

1. All correctness and factual repairs, data-history audit, current paired quality and reconstructible evidence.
2. Four main model/drafter pairs; representative three-seed checks.
3. At least one well-matched cheap trainable alternative per drafter family, with the closest frozen-steering/independent-drafter comparison included wherever it can be faithfully implemented. If an important comparison remains missing, state that limitation and reassess the central claim.
4. The essential normalization, data-versus-updates and capacity controls, with reduced development sweeps rather than many extra variants.
5. Existing task breadth including negative cells and one clean second-family replication.
6. Isolated memory, a complete rewrite, automatically rebuilt figures and an anonymous reproducible artifact.

A production-engine port is conditional in this focused schedule. If it cannot be completed, remove production-serving claims and explicitly limit the paper to the validated runtime. Do not delete an inconvenient result or retain an unsupported general claim to make the schedule work. E13 forecast, E15 drift/RL, broad new transfer adapters and additional architecture exploration are postponed.

**This reduced version does not automatically satisfy the current full-plan readiness checker.** Its ledger still requires the full strong scope, including E12. Before executing a reduced scope, record a dated amendment describing exactly which claims and deliverables change. Do not quietly change a required task to optional.

| Date window | Must produce |
|---|---|
| Sept 5–8 | Decoder diagnosis and fixes, correct interface equations/export, frozen data protocol |
| Sept 9–12 | Main current-output comparisons, cheap-baseline pilots, seed checks |
| Sept 13–16 | Selected final baselines, essential ablations, breadth and second-family evidence |
| Sept 17–18 | Evidence-supported abstract; settle final paper scope |
| Sept 18–21 | Freeze runs, generate tables/figures and rewrite |
| Sept 22–24 | Independent claim checking, PDF inspection, clean reproduction and anonymous package |

This deadline schedule requires strong daily focus, overlapping CPU analysis/writing with GPU runs, and no prolonged correctness blocker. At six productive hours per day, 90–130 hours represents 15–22 working days before additional queue/dependency delays. At eight hours per day, it represents about 11–16 working days; the remaining days provide limited buffer. Assistance can reduce mechanical work, but it cannot remove the need to inspect results and resolve uncertain scientific conclusions.

## Biggest schedule uncertainties and stop points

| Uncertainty | Decision point |
|---|---|
| Unexplained target-output differences | If the first 16–24 active debugging hours do not identify and validate a repair, revise the deadline forecast immediately. Do not launch the full matrix on a suspected incorrect decoder. |
| Cheap baseline needs an unsupported integration | Run a compatibility pilot first; after one working day, estimate the actual port before committing the experiment budget. Missing a central baseline weakens the claim, so scope changes need explicit reasoning. |
| Production engine incompatibility | After one to two focused days without a functioning minimal path, defer production claims for the deadline version. |
| Data exposure or near-duplicate problems | Freeze the new evaluation only after the history audit. New data preparation and scoring may add several days. |
| Queue availability | Today's idle nodes are not a reservation. One historical job waited about 3.5 hours; another started immediately. Some waits were caused by dependencies/resource limits. These observations do not establish a queue-time distribution. |
| Scientific result contradicts the proposed advantage | Analyze and report it. A methodological redesign is a new estimate, not an ordinary rerun. |

Keep at most four GPUs allocated across all RelaySpec jobs. A four-worker job uses the entire authorized budget; two simultaneous four-worker jobs exceed it. Prefer one such job with requests distributed across the four GPUs while CPU analysis and writing proceed independently. Do not keep GPUs allocated solely for writing or idle planning.

## Next estimate update

First complete the reuse inventory and subtract all already-valid work from each task. Then use the smallest necessary diagnostic and cheap-baseline pilot to update unresolved runtime estimates. A new main paired run is needed only when its existing evidence is invalid or insufficient for the intended claim. Recompute the remaining matrix and calendar from this inventory; until then, quoting a precise completion date for every planned extension would be misleading.
