# Controlled mapper scaling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute the requested fitting and capacity study with bounded exploratory trials and auditable final paper evidence.

**Architecture:** Extend existing training and benchmark paths; preserve historical snapshots and the four-GPU allocation ceiling. Keep fast development trials separate from full paired evaluations. Use the detailed scientific matrix in `2026-09-05-relayspec-scaling-autoresearch.md` as the required study ledger.

**Tech Stack:** PyTorch DDP, Python, Slurm on Turing node07, JSON/YAML, pytest, LaTeX.

---

## Live inventory at 21:57 IST

Continuous 27583 and unfitted pilot 27673 completed. Data64 27675 running; data128 27676 and data256 27677 queued. All use immutable remote snapshots. Collectors were absent and their status files stale; restart with the documented scorer dependency overlay. Existing code/evidence baseline: 208 CPU tests passed.

## Task 1: Recover and explain completed evidence

Files: existing `scripts/watch_controlled_jobs.py`, `scripts/summarize_fitting_search.py`, `reports/controlled-scaling-20260905/`, `reports/unfitted-controls-20260905/`.

1. Restart both collectors with `PYTHONPATH=src:vendor/qwen-score-deps`.
2. Verify all checkpoint/completeness gates and rescore existing outputs.
3. Run `scripts/summarize_fitting_search.py` and retain uncertainty, development exposure and seed qualifications.
4. Record runtime by training, model loading and generation arm. The 512-example job took 22m34s while fitting took 98s; slow AR evaluation dominates.

## Task 2: Capacity and regularization implementation

Files: modify `src/relayspec/relay.py`, `src/relayspec/losses.py`, `scripts/train_relay.py`, `scripts/benchmark_relay.py`, `scripts/benchmark_eagle3.py`; test `tests/test_mapper_scaling.py`.

1. Add failing tests for factorized-vs-collapsed equivalence, matched MLP parameter counts, checkpoint reconstruction, incompatible architecture options and explicit L2 gradients.
2. Implement `factorized_rank`, preserving existing checkpoint behavior by default. Dense count d*k; factorized and MLP count h*(d+k).
3. Implement `l2_weight / 2 * sum(trainable_weights**2)` separately from AdamW decay; reject simultaneous penalties in controlled studies and log both terms.
4. Thread architecture metadata through save/load for both drafter families.
5. Run focused tests followed by normal lint and CPU suite before deploying the snapshot.

## Task 3: Fast trials with enforceable budgets

Files: create `src/relayspec/scaling_trials.py`, `scripts/run_mapper_trial.py`, `slurm/mapper_trial.sbatch`, `configs/submission/scaling/`; test `tests/test_scaling_trials.py`.

1. Build immutable trial JSON: hypothesis, parent, architecture, N, seed, exact updates, explicit regularization, validation role, evaluation budget and source hashes.
2. Pilot: 16 updates, 8 paired development requests, 128-token cap; directional: 1024 updates, 16 paired requests, 256-token cap. Hard 540-second process timeout and 10-minute Slurm limit include loading.
3. Preserve full 128-request/2048-token historical protocol for the existing queued minimum-data cells. New search trials use the short protocol. Do not shorten a running experiment or silently pool protocols.
4. Record capped trial outputs as throughput/acceptance diagnostics; they cannot establish full-answer quality. Final paired evaluations remain a separate promotion step.
5. Submit dense control, factorized-512 and MLP-512/L2 infrastructure pilots after current allocations finish; chain dependencies so at most four GPUs run. No blind full-grid submission before gates pass.

## Task 4: Validation and larger data

Files: extend `scripts/train_relay.py`; create manifest builder and validation helpers with tests.

1. Freeze disjoint fitting-validation, decoding-development and confirmatory roles with full problem/solution hashes. Data already inspected remains development-exposed.
2. Evaluate fixed training/validation diagnostic subsets at saved checkpoints; log unregularized relative MSE, cosine/norm errors and token counts.
3. Build nested 8192/16384/32768 actual records from a documented eligible source; block levels unavailable in the current 4096-record manifest. Audit exact and near overlap before fitting.
4. Measure feature-cache cost/storage and test batching equivalence before accelerating repeated fits. Cache only frozen features with complete provenance.
5. Execute the 35-cell core, 16 extension cells, fixed-exposure versus fixed-pass panels, regularization contrasts, seed confirmation and family replications from the scientific plan. Report unsuccessful outcomes too.

## Task 5: Remaining submission evidence and bounded autoresearch

Files: `configs/submission/plan.json`, existing scoring/diagnostic scripts, paper evidence builders and manuscript.

1. Rescore saved code outputs, finish output-divergence diagnosis and retain conversation-quality gaps explicitly.
2. Run matched-budget drafter adaptation, fitting seeds, selected task/family and serving controls. Update costs from measured GPU-seconds and parameter counts.
3. After fixed baselines, log every single-change hypothesis and falsifier; use short trials, never select on confirmatory data. Stop each branch after three failed trials or two rounds without useful improvement.
4. Promote only validated evidence, regenerate tables/figures and claim map, compile and inspect the final PDF. Required completion means scored, audited evidence, not successful Slurm exit or queued work.

## Execution policy

User requested autonomous implementation and parallel use of four GPUs. Continue without an execution-choice prompt. Use isolated branch `research/scaling-autoresearch-20260905` in `/home/aryamavmurthy/work/RelaySpec-scaling`. Preserve user untracked work. Full study completion remains pending until the above artifacts exist; no claim that a universal optimum or paper readiness has been achieved.

## Execution amendment: efficiency and data availability

The user prioritizes robustness, then time, and requires a <=10-minute pilot
for every new path. The first mapper pilots measured 64–72 seconds. Before
expanding the full grid, implement two cost reductions with their own pilots:
(1) frozen-feature caches and four independent cached fitting trials at once,
with batch-four record weighting validated against DDP; (2) multiple maps in
one rotated paired decoding campaign so AR/native/source are measured once
per request rather than once per map. Include feature extraction, validation,
I/O and fitting costs separately. Do not reduce the declared scientific grid.

The expanded panel now has real 32768-example nested manifests from a pinned
NuminaMath source shard and a fixed 1024-example fitting validation split.
Historical MATH-only runs remain separate. The builder rejects wrong source
file hashes and audits problem/solution lexical overlap; final confirmatory
decoding data and semantic/template independence remain unresolved.

Execution details, invalidated diagnostics, exact jobs and next steps are in
`reports/mapper-scaling-20260905/EXECUTION.md`.

## Full cached matrix execution

`configs/submission/scaling/matrix-v1/matrix.json` freezes 51 primary settings,
42 nonzero regularization settings, and 3 useful seed-confirmation cells to
fill four-GPU batches without dummy duplicate work. The zero-penalty arms
are the matching primary cells. Each trajectory saves both 8192 updates
(fixed 32768 presentations) and N updates (four passes), with shared checkpoints
when the two budgets coincide. Training diagnostics use min(N,256) records,
validation uses the fixed 1024-record split; early training-pool diagnostics can
include records not yet visited. The larger-data panel uses Numina rather than
the historical MATH fitting source, and must remain separately labeled.

The resource pilot tests widths64 and4096 in both families at16updates on the
full streaming cache and then performs shared paired decoding. It has a10minute
Slurm ceiling and540second process bound. Full batches are gated by this exact
cache's successful resource pilot and have up to75minutes for the largest
32768-update trajectories. Fit-only jobs do not include repeated AR generation.
Batch grouping uses declared approximate fitting costs; actual stage timings
and all four GPU traces are retained. Full batches are submitted in stages,
with correctness, elapsed time and memory reviewed before further promotion.

Full-cache job27723 follows the directional trial chain and uses source727d370.
The first all-repo software check yielded239passes plus3 manuscript failures
due to absent compiled .aux artifacts; the separately invoked evidence tests
added3passes. Thus242 software/evidence tests pass, while the3 compiled-paper
audit tests remain pending a manuscript build. No submission readiness is claimed.

## Epoch and optimization amendment (user steering, 23:38 IST)

The user explicitly requested more epochs and more data for MLP/regularized
models before judging them. `matrix-v2` preserves all v1 cells and budgets and
adds checkpoints after epochs1/2/3/4 and later powers of two within each budget.
At N32768 this includes four full epochs; at N512 the fixed-exposure endpoint
uses64epochs. Per-update logs/checkpoints record epochs separately from distinct
records. The final optimizer, mapper and RNG state are preserved for exact
continuation, so improving endpoints can be extended without refitting.

Assess convergence before interpreting nonlinear capacity: if validation keeps
improving at the endpoint, extend matched linear/MLP trajectories to8epochs and
then16 as needed, retaining their original endpoints. If train and validation
remain poor or optimization is unstable, use matched learning-rate checks
(1e-4,3e-4,6e-4,1e-3), first under the10minute pilot/directional ceiling,
then full trajectories for useful directions. A falling training loss with
rising validation loss motivates the declared regularization/data comparison.
Neither an early low-throughput result nor expressive capacity alone establishes
generalization or a disadvantage of MLPs. These convergence checks support the
fixed study; hypothesis-driven autoresearch still follows the fixed baselines.

The capacity pilot is replaced before it runs so the epoch logging and optimizer
round-trip checks are exercised within its10minute limit. No full fits have
been submitted against the v1 matrix. The original declaration remains for
provenance.

## User scope update, 6 September 2026

Further large-data scaling is PAUSED at the user's request. Do not restart
the32,768-example job27728 or submit new large-data/four-epoch replicas.
Preserve the completed128--8192 fitting curves and the verified full cache.
The partially run27728 is a stopped attempt, not a completed scientific cell.

Current priorities are capacity, matched linear/MLP comparisons, explicit L2
and AdamW decay, convergence/learning-rate checks, seeds, model/family
replication, actual decoding/quality and the other pending paper baselines.
Use512 and2048 distinct fitting records for the focused capacity and
regularization study. Existing8192 outcomes can be analyzed, but further
large-data expansion is deferred. This user change supersedes the earlier
requirement to execute all32768 cells now.

Cache-access pilot27730 is an implementation speed/equivalence check using
4096 records for mapped I/O and512 for GPU residency, not a new32k data
scaling experiment. Keep its10minute ceiling; its results may accelerate
the remaining smaller-data fits.


## Focused execution after the large-data pause

The exact smaller-data declaration is `configs/submission/scaling/matrix-focused-v1/matrix.json`.
It contains30 capacity settings at N512/2048 (dense and linear/MLP widths
64/128/256/512/1024/2048/4096), reuses four completed width512 fits from27726/27727,
and declares42 separate nonzero L2/AdamW arms plus four dense seed confirmations.
Thus72 new fits remain in18 four-GPU batches. Submit in measured stages, not a
blind queue. No new8192/32768 data cells are in this declaration.

Cache-access pilot27730 completed in2m05s and passed bit-identical weights,
optimizer states and all logged losses. Mapped streaming improved loop time
106.73s to68.87s (1.55x); GPU residency improved19.73s to13.41s (1.47x).
New optimized fits require the same-cache exact-equivalence gate and capacity
gate. New batch scripts default to540seconds/10minutes, including fitting,
validation and checkpoint verification. Full raw pilot outputs are collected.

The next decoding check uses eight completed factorized/MLP512 checkpoints,
N128/512/2048/8192, with shared AR/native/source controls on16 MATH requests
capped at256tokens. This analyzes existing data and does not resume large-data
fitting. It remains a development throughput/acceptance diagnostic, not a
full-answer quality result. CPU validation:16 targeted tests and lint passed.
