# Mapper scaling execution, 5 September 2026

Status: active research; the paper and full requested study are not complete.
Code lives in branch `research/scaling-autoresearch-20260905` at
`/home/aryamavmurthy/work/RelaySpec-scaling`. Historical result collection lives
in `/home/aryamavmurthy/work/RelaySpec/reports/`.

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
