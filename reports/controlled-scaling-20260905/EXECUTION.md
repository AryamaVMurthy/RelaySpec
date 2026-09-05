# Controlled scaling execution

Started September 5, 2026 on Turing node07, using four NVIDIA L40S GPUs.
The maximum simultaneous GPU allocation is four across the entire job chain.
GPU jobs are batch jobs and continue after the interactive session ends.

## Completed checks and current queue

| Job | Work | State at launch review | Wall-time limit |
| --- | --- | --- | --- |
| 27545 | EAGLE-3 8B breadth against AR and source reuse | Completed in 1 h 32 min 59 s, all 2,490 rows | 4 h |
| 27578 | Controlled fitting pilot, checkpoints at 8 and 16 updates | Passed in 1 min 37 s | 10 min |
| 27546 | EAGLE-3 14B breadth against AR and source reuse | Running on all four GPUs | 4 h |
| 27580 | 512 distinct records, 1,024 fitting updates | Queued after 27546 | 90 min |
| 27581 | 1,024 distinct records, 1,024 fitting updates | Queued after 27580 | 90 min |
| 27582 | 2,048 distinct records, 1,024 fitting updates | Queued after 27581 | 90 min |
| 27583 | One continuous fit, checkpoints at 128/256/512/1,024/2,048 updates | Queued after 27582 | 4 h |

All new long runs required the successful pilot. Independent data cells use
`afterany` for serialization, so a failed cell does not block an unrelated one.
Such a failure remains a failure and is recorded by the collector. No failed
result is promoted. The pending 14B job was briefly held to insert the pilot,
then released. It has an `afterany` dependency on the pilot, since breadth does
not depend on the new trainer.

Expected remaining time is approximately 6–9 hours, including 14B breadth and
4–6 hours for controlled scaling. Queue delays are additional. The remaining
configured job limits sum to at most 12.5 hours, or 50 allocated GPU-hours,
before subtracting time already spent in the running 14B job. These are upper
limits, not expected durations. No additional main MATH-500 rerun was submitted.

## What the new experiments hold fixed

The data-budget runs use nested prefixes of the existing 4,096-record fitting
manifest. All use DFlash 4B → Qwen3-8B, seed 1729, the selected normalized linear
interface and relative-error objective, 1,024 updates and four records per
update. Smaller sets cycle in their original order. They therefore receive the
same 4,096 presentations while distinct data changes.

The continuous run uses the same ordered 4,096-record manifest in one process.
AdamW state is never reset between checkpoints. Early checkpoints have seen
512, 1,024 and 2,048 distinct records, respectively. The 1,024-update checkpoint
has seen all 4,096 records once, and 2,048 updates gives two passes. Every
checkpoint records available distinct records, distinct records actually seen,
presentations, fitting time, map weights and optimizer steps. The
1,024-update checkpoint supplies the 4,096-record cell of the data-budget
comparison, avoiding a duplicate fit and evaluation.

Each checkpoint is evaluated on the same fixed, seed-selected 128 MATH-500
requests, with a 2,048-token cap and one warmup. Every run measures AR, native
target DFlash, source reuse and RelaySpec in rotated order. The pilot instead
uses eight requests and a 64-token cap. Pilot speeds are infrastructure checks,
not paper results. The 128-request set is an analysis set with prior development
exposure, not a newly blinded confirmatory set.

## Validation and measurement

The pilot executed real matrix work on all four GPUs, trained for 16 updates,
and saved two different maps. Both saved AdamW step counters were checked.
Both maps reloaded through the actual decoder. Each produced all 32 planned
method/request rows with finite positive timing. Proposed-token acceptance
counts also passed. The CPU optimizer test verifies that saving a checkpoint
does not change the RNG or the uninterrupted AdamW trajectory.

Same-vocabulary DFlash runs now record actual proposed block positions and
accepted draft positions for each verification cycle. Acceptance excludes the
target-supplied position and counts verified positions before final EOS/output
cap trimming. It is reported separately from delivered output length and mean
cycle progress. Instrumentation adds only integer/list bookkeeping.

The CPU check passed all 206 non-manuscript tests. All three manuscript
checks also pass. New results are staged outside the published paper evidence
directory until they are reviewed for promotion. Existing checkpoint loading remains compatible. The pilot
also exercised the new checkpoint format, rank traces, nested data accounting,
benchmark completeness gates and acceptance denominators.

## Collection and analysis

`jobs.json` records the exact remote and local paths. The collector polls only
these seven jobs, downloads completed records, and computes absolute throughput,
AR-relative throughput, paired 95% intervals, native throughput retention where
measured, progress and proposed-token acceptance. It scores saved math outputs
using the existing Qwen scorer and dependency overlay, whose recorded hashes
are verified. MT-Bench intervals resample conversations, not individual turns.
Scoring runs locally on CPU so GPUs can start the next experiment immediately.
Code and conversation quality need their own evaluators and are not inferred
from generation success or math scores.

Run the bounded collector from the repository root:

```bash
PYTHONPATH=src:vendor/qwen-score-deps .venv/bin/python scripts/watch_controlled_jobs.py
```

It stops after all jobs have been collected or failed, or after 24 hours. It
does not submit, cancel or retry GPU jobs. Collection errors are logged, and
each analysis subprocess has a 20-minute limit. `collector-status.json` records
live status. Generated results await scientific review before paper promotion.
The large map checkpoints are preserved in the remote home workspace. Optimizer
snapshots remain on scratch. The collector excludes `.pt` files from Git-sized
evidence transfers. Raw records, source snapshots and compact metrics are saved.

## First completed breadth finding

EAGLE-3 8B RelaySpec throughput relative to paired AR is:

| Workload | Throughput ratio | Paired 95% interval |
| --- | ---: | ---: |
| GSM8K | 2.403 | [2.348, 2.462] |
| HumanEval | 1.832 | [1.784, 1.889] |
| MBPP | 1.828 | [1.794, 1.868] |
| MT-Bench | 1.160 | [1.088, 1.247] |

GSM8K scores are 119/128 for RelaySpec and 120/128 for AR. These scores evaluate
the saved timed outputs. Full task-level metrics and uncertainty are in the
completed run's `analysis.json`. The manuscript is not automatically rewritten
from this interim result.
