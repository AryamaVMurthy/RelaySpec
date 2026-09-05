# Mapper scaling execution, 5 September 2026

Status: active research; the paper and full requested study are not complete.
Code lives in branch `research/scaling-autoresearch-20260905` at
`/home/aryamavmurthy/work/RelaySpec-scaling`. Historical result collection lives
in `/home/aryamavmurthy/work/RelaySpec/reports/`.

## Latest state, 23:50 IST

- Full frozen-feature cache27723 completed in12m22s:32,768 train records,
  1,024 validation records and292,810,789,056bytes. Cache SHA-256 is
  a2fd012df1da5554f8e211cf0b60aec098d3754d866bc50e9623f3117f43f8e5.
- Epoch/continuation capacity pilot27725 passed in2m17s. It exercised widths
  64 and4096 for both linear and MLP models, streaming the full cache, fixed
  1024-record validation, BF16 equivalence and saved-optimizer round trips.
  Pilot27724 was cancelled while still pending and consumed no GPU time.
- Full MLP512 data trajectories are running as27726 (primary-04), one per GPU
  at N128/2048/512/8192. Matching linear512 trajectories27727 follow. Both
  use immutable source13ab419 and matrix-v2, preserving explicit epoch
  checkpoints and both budget panels. Job ledger:matrix-wave1/jobs.json.
- Cached/full/scoring collectors run detached on the workstation. Revalidate
  their PIDs and Slurm before relying on these observations.

## Verified work

- Commits `86b1557`, `194ba65`, `308d32d`: factorized linear maps, matched-width
  MLP loading in both benchmark families, explicit L2 distinct from AdamW decay,
  training/validation diagnostics, bounded pilots/directional trials, larger
  nested data builder, and exact-task-subset EvalPlus wrapper.
- 228 CPU tests passed (3 manuscript tests excluded), with lint/format checks.
  This establishes software checks, not paper readiness.
- Continuous run 27583 and unfitted pilot 27673 recovered and scored.
  Continuous 128/256/512/1024/2048 updates: 3.65/4.34/4.67/4.90/5.03x AR;
  each scores 105/128 on development-exposed MATH requests.
- Data64 job 27675 completed and scored: 3.55x AR, ~72.8% of observed-best
  mapper throughput, 105/128 correct. It fails the descriptive 95% threshold.
  The fixed-compute 512/1024/2048/4096 cells remain within 5% of observed best.
- Data128 27676 completed and was collected/scored: 4.10x AR, ~83.7% of
  observed-best throughput, 105/128 correct. This also fails the 95% threshold.
- Original pilots 27685–27689 completed in roughly one minute each. Their
  training/checkpoint/decoding outputs are valid infrastructure evidence, but
  fitting-validation metrics are INVALID because they omitted DFlash's final
  normalization. Do not interpret them as overfitting evidence.
- Correction `194ba65` applies the exact released `draft.hidden_norm` in
  diagnostics. Pilots 27694–27698 use a separate immutable v2 snapshot.
  All five corrected pilots passed checkpoint/completeness/diagnostic checks in
  62–69 seconds. Directional jobs 27711–27715 are queued behind data256; each
  uses 1024 updates and the fixed <=10-minute trial budget.
- Data256 27677 follows the corrected pilot chain. No active benchmark was
  cancelled. Every GPU job uses all four physical L40S GPUs on node07.

## Time and GPU policy

Historical 128-request evaluations generate four arms up to 2048 tokens.
The 512-example fit took 98 seconds in a 22m34s allocation; AR generation alone
used 11m26s on the busiest worker. Search pilots use 8 requests / 128 tokens;
directional trials use 16 / 256 and 1024 fitting updates. Slurm limit is 10 min;
process timeout 540 seconds. Capped outputs are not final task-quality evidence.

Next efficiency work: reuse frozen feature caches after an equivalence pilot,
fit independent candidates in parallel within one four-GPU allocation, and
benchmark multiple candidate maps in one paired campaign with a shared,
rotated AR/native/source reference. Avoid repeating identical AR generations
for every candidate. Do not splice historical timings into new paired claims.

## Expanded data

`data/scaling/manifests-pinned/` contains nested 128/512/2048/8192/16384/32768
training sets, fixed 1024 validation and 256 training-diagnostic records.
Source is immutable NuminaMath-CoT revision
`9d8d210c9f6a36c8f3cd84045668c9b7800ef517`, train shard 0/5, restricted to
`cn_k12`, `synthetic_math`, `orca_math`. This is a separate data-source panel,
not a continuation of the historical MATH-only curve. The pinned first shard
contains 119604 records in these three strata before filtering.

The builder checks immutable source-file hashes, excludes all original MATH
test problems/solutions, historical fitting problems/solutions and current
benchmark prompts, then excludes validation from training. It uses normalized
exact matches and 5-shingle Jaccard >=0.6, and removes duplicate problems.
For the full set it excluded 17 exact matches, 527 near matches and 252 duplicate
problems. Lexical checks do not establish semantic/template independence.
Source mixture, counts, file hashes and code hashes are in `data/manifest-gate.json`.
The 32-record preparation pilot took 4.3 seconds; full preparation took 50 seconds.
Raw parquet files and 90 MB of generated manifests stay outside Git.
Remote data destination:
`/scratch/aryama.murthy/factorspec-runtime-20260826/relayspec/scaling-data/numina-v1`
(compute-node view; login uses `/scratch/node07/aryama.murthy/...`).
Verify rsync completion and hashes before feature extraction.

## Code-quality scoring

CPU-only pilot sequence on node01:
- 27692 failed: original scorer environment lacked `datasets`.
- 27704 failed: dependency transfer had not completed; `libarrow_substrait`
  was unavailable. Transfer is now complete. Do not use this run.
- 27705 failed: official EvalPlus requires all tasks; a two-task pilot is a
  subset. No quality result is claimed from this attempt.
- 27707 passed the corrected subset wrapper in 2m14s under a 10-minute limit, using
  two HumanEval and two MBPP tasks for all three saved EAGLE-3 methods.

The wrapper preserves every official base/plus test for selected IDs, rejects
unknown or duplicate IDs, uses a separate subset ground-truth cache key and
records scorer/data/sample hashes. It adds no missing-task placeholders.
Pinned upstream EvalPlus commit: `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`.
CPU source snapshot: `/home/aryama.murthy/relayspec-scoring-20260905`.
Dependency overlay: `/home/aryama.murthy/relayspec-eval-data-deps-20260905`;
package versions are in `scoring-data-dependencies.txt`.
Full CPU scoring jobs 27708 (8B) and 27709 (14B) are running with eight CPUs
each on node01. No GPU generation rerun is used. Conversation quality remains
a separate unresolved task.

## Next actions

1. Revalidate live Slurm jobs and durable collector PIDs (do not trust stale
   status files). Collect 27677 when finished and update the minimum-data report.
2. Collect and interpret directional trials 27711–27715 after data256 finishes.
   Corrected pilot gates already passed; directional outcomes remain pending.
3. Collect full code-scoring jobs 27708/27709; preserve failures and provenance.
4. Implement and pilot cached feature fitting with record-weighted masked
   batching equivalent to global batch four, then shared-reference multi-map
   decoding. Cache extraction cost must be reported separately from fitting.
5. Execute the crossed scientific matrix, regularization, larger-data/longer
   training trajectories, seeds, family/task replication, cheap-adaptation
   baseline, and bounded autoresearch from the detailed plan.
6. Promote only audited evidence, regenerate figures/tables and manuscript,
   compile/inspect the PDF and perform the complete submission audit.

No universal minimum/optimum, completed autoresearch campaign, or paper-ready
status has been established.

## Verified update, 23:24 IST

- Commit b7efe32 implements hashed frozen-feature extraction, independent cached
  fitting on four GPUs, and shared-reference multi-mapper decoding. 239 CPU tests
  passed (3 manuscript tests excluded). End-to-end pilot 27722 passed in 1m51s:
  64 train / 16 validation records, four simultaneous 16-update fits, cache
  integrity and BF16 gradient-equivalence gates, and exact duplicate-mapper
  decoding/acceptance isolation. It is plumbing evidence, not a capacity result.
- Full data256 job 27677 completed in 22m41s; fitting took 97.93s. It achieved
  4.542x AR for decode-only timing (4.512x for end-to-end request timing). The complete single-seed historical curve's smallest tested set
  within 5% of the best measured throughput remains 512. Independent confirmation
  is outstanding. All seven data sizes are collected and scored.
- Dense directional 27711 completed in 3m18s (97.04s fitting); factor512 27712
  completed, with MLP and regularized directional trials progressing.
- CPU code quality jobs 27708/27709 completed and were collected. CODE_QUALITY.md
  and code-quality-ar.json contain official EvalPlus base/plus AR-relative paired
  outcomes and uncertainty. 8B MBPP plus loses four passes; 14B gains two.
- Full cache extraction is now a separate gated stage. Pilot estimates about
  300 GB for 33,792 records; scratch has about 14 TB free. Extraction cost and
  model-loading wall time are logged separately and charged to the study.

## Epoch-aware fitting and paper update

All five 1024-update directional jobs27711-27715 completed in2m58s-3m18s
and were collected. Their source-hashed aggregate is directional-summary.json.
Dense/factor512/MLP512 end-to-end speedups are4.610/2.867/2.526x AR on16
development requests capped at256tokens. L2=1e-6 did not improve the fixed
validation objective. These are early-budget single-seed outcomes, not a
verdict on MLP capacity or generalization.

User steering added explicit epoch1/2/3/4 and later power-of-two checkpoints
within the two fixed budget regimes (matrix-v2). Optimizer and RNG state
are preserved at each trajectory endpoint for further fitting when justified.
Convergence/learning-rate checks remain required before interpreting poor MLP
results. Source13ab419 includes these changes.

The manuscript appendix now includes the completed seven-point historical data
curve, five-point continuous curve, and8 AR-paired code-quality comparisons.
New tables/figure are generated with raw-source hash checks. The draft compiled
to20pages with main text ending onpage9 and42 resolved citations. Changed
pages16-17 were rendered and inspected. The full color/grayscale visual-review
record is stale and remains pending the final paper build. New Numina capacity
results, seeds, family/task transfer and autoresearch are still outstanding.

## Large-data batch queued, 23:56 IST

Job27728 follows27727 and fits dense, MLP2048, factorized4096 and MLP4096
on32,768 distinct records, each through four epochs (32,768updates).
The same trajectory includes the8192-update fixed-exposure checkpoint.
Its ledger and independent collector are matrix-large-data/jobs.json.
The full-cache path, source13ab419 and successful capacity pilot27725
remain fixed. The current MLP512 job27726 has reached7424/8192updates
on its last, streaming-data worker; the three smaller-data fits are ahead.
Do not infer all matrix cells are complete from these initial batches.

Measured execution suggests grouping remaining batches by cache I/O as well
as parameter count: small preloaded-data workers finish before the larger
streaming worker. Preserve scientific cell definitions while improving
scheduling. A GPU-resident small-data cache or work queue needs its own
short equivalence/resource pilot before changing the fitting path.

## First full MLP data curve completed, 23:58 IST

Job27726 passed in8m08s. Each MLP512 fit used8192updates/32768presentations.
At N128/512/2048/8192, fixed1024-record validation objective was respectively
0.37380/0.30818/0.28776/0.28184. The corresponding epochs were256/64/16/4.
Training diagnostics were0.24254/0.26714/0.28123/0.28295, using min(N,256)
training-pool records. These single-seed fitting results support further
large-data study, but do not establish full-answer speed or quality gains.
Source-gated checkpoints/trajectories are in initial-mlp-data-curve.json.

The smaller fits' loops took123-147seconds; the N8192 streaming loop took
440seconds, including375seconds recorded in input I/O. This measured cost
should guide the next scheduling/cache optimization pilot. Matching linear
job27727 is now running and large-data27728 remains queued behind it.
