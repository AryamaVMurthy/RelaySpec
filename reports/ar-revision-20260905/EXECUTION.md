# AR-primary revision and execution record

Date: 5 September 2026. Public repository: AryamaVMurthy/RelaySpec.

## Completed evidence reused

The four main runs are complete. They were recovered, not regenerated. Their
recorded four-GPU durations sum to 7 h 42 min. That cost is already paid and is
excluded from the next 24-hour budget.

| Drafter / target | Historical job | AR tokens/s | Relay tokens/s | Relay / AR | Native throughput retained |
|---|---:|---:|---:|---:|---:|
| DFlash / 8B | 27357 | 37.37 | 186.49 | 4.99× | 90.3% |
| DFlash / 14B | 27325 | 22.88 | 116.88 | 5.11× | Not measured in this run |
| EAGLE-3 / 8B | 27396 | 36.15 | 84.92 | 2.35× | 90.2% |
| EAGLE-3 / 14B | 27397 | 22.88 | 59.18 | 2.59× | 89.1% |

Each main method has 500 request records. DFlash breadth jobs 27498 and 27499
supply all eight AR-paired target/task cells. Six completed 128-request fitting
studies and the earlier paired block lengths 8, 16 and 32 are reused. Historical
source-reuse breadth and isolated memory remain separate experiments.

`raw/` preserves outputs, configurations, manifests and the available original
source snapshots. `paper/iclr2027/generated/ar_evidence.json` records derived
statistics and input hashes. The asset builder checks method pairing and score
identity. Throughput is summed generated tokens divided by summed request time.
It is not the inverse-time ratio when output lengths differ.

Math and GSM8K scores were recomputed from these saved completions using the
existing Qwen evaluation code. `scorer-provenance.json` identifies the recovered
scorer by content hashes. Its upstream Git revision was not recoverable from the
copied directory. The callable is `build_math_scorer` in
`scripts/aggregate_results.py`. No newly generated code-quality scores are
inferred from historical programs.

## Executed and running work

All jobs use at most four L40S GPUs on Turing node07. The two full runs are
serialized by an `afterok` dependency. The isolated workspace is
`~/relayspec-day1-20260905` and outputs have unique job-numbered directories.

| Job | Experiment | Recorded state at this update | Duration / allocation |
|---|---|---|---|
| 27533 | DFlash 8B, eight selected diagnostic prompts | Completed | 1 min, four GPUs |
| 27534 | EAGLE-3 8B, eight selected diagnostic prompts | Completed | 59 s, four GPUs |
| 27541 | EAGLE-3 8B breadth pilot | Generation completed. Post-run gate passed after dependency fix on saved outputs | About 1 min, four GPUs |
| 27543 | EAGLE-3 14B breadth pilot | Completed, completeness gate passed | 58 s, four GPUs |
| 27545 | EAGLE-3 8B full breadth | Running at the latest check | Four-hour job limit |
| 27546 | EAGLE-3 14B full breadth | Queued after successful 27545 | Four-hour job limit |

Pilots have a ten-minute Slurm limit and a 540-second process timeout, including
loading. Job 27541's GPU generation produced all 30 expected method/request rows.
Its original postprocessing imported an unavailable plotting dependency. The gate
was made independent of plotting, then passed on the same saved records. Those
valid outputs were not regenerated. Both breadth pilots cover ten requests
from eight initial records, including two-turn conversations.

The first diagnostic is explicitly instrumented and is not throughput evidence.
Compressed target-call traces are in `diagnostics/`. In eight selected DFlash
cases, native, source and relay drafting select the same first differing token
from AR. At that position the leading BF16 target scores tie or reorder, with
observed gaps of zero or 0.25. This is evidence about those cases. Controlled
precision and call-shape replay is still needed to test the cause and coverage.

## Paper decisions

- Primary comparisons are absolute throughput and speed relative to AR measured
  in the same run. Source reuse answers the narrower source-removal question.
- Native throughput retention is relay tokens/s divided by native tokens/s. It
  is not a percentage of speedup above AR.
- Training counts are 4,096 additional relay records versus approximately 800K
  in the published DFlash recipe and 532K in the published EAGLE-3 recipe. The
  EAGLE count is not an audited count for the evaluated DeepSpec checkpoints.
  Record lengths, objectives and inherited drafter work differ.
- Existing scaling changes distinct data and updates together. The 8,192-point
  means two presentations of each of 4,096 distinct records. It is not a
  continuous training trajectory or 8,192 independent examples.
- Report mean progress per cycle, its counting convention, and position-wise
  survival. A proposed-token acceptance fraction needs the actual proposed-token
  denominator and must exclude target-supplied progress.
- Preserve complete workload matrices. Explain smaller source-relative gains
  through accepted progress and cycle cost without asserting causality from
  descriptive comparisons alone.

The updated plan is `docs/plans/2026-09-05-relayspec-next-24-hours.md`.
Running jobs and proposed experiments are not manuscript results.
