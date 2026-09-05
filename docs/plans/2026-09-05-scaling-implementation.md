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
