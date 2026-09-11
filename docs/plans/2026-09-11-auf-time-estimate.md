# AUF campaign time estimate

**Superseded budget:** The finalized study now uses three main loss variants, one principal fit seed, three principal timing measurements, and additional training seeds only for Q8/L3 robustness. Fusion-LoRA is a targeted ablation. The10–18day estimate and five-variant3×3 arithmetic below describe the earlier larger scope, not the finalized campaign. See `2026-09-11-auf-vllm-study.md`, Section8, for current counts.

Estimated on 2026-09-11 from the written experiment matrix and archived run timings. No AUF GPU run has started. All future times below are planning estimates, not measurements. Assume four comparable L40S GPUs available continuously, matching local model assets, a resolved target checkpoint/adapter, and no scheduler downtime. Calendar estimates begin when implementation starts; queue or missing-input delays are additional.

## Historical measurements

Source: `reports/transfer-reproduction-20260907/FULL_REPRODUCTION_RESULTS.md`, checked against `full16384-results/fit-summary.json`.

| Completed Qwen ZIP stage | Observed elapsed time |
|---|---:|
| Generate 16,384 training records | 96m25s |
| Capture paired sampled features and validate | 51m58s |
| Fit 16,384 records for three epochs | 14m28s invocation; 14m25s epoch compute |
| Training/audit/export/eight-prompt job | 15m50s |
| AR/native/mapped comparison,128 prompts,2048 cap | 31m47s |
| Additional speculative timing repeat | 7m43s |

The generation job completed the records but failed its historical-hash comparison; those records were explicitly treated as a new realization. These timings are different stages with their original resource assignments, not four-GPU-parallel estimates for every stage.

The benchmark generated124,899 actual output tokens per method, or975.8/request. Throughput was26.05tok/s for AR and167.83tok/s for mapped decoding. Archived Llama and cross-family request rates were114.93 and59.41tok/s in a different runtime/precision mode; these give scheduling sensitivity examples only, not predicted new vLLM performance.

For124,899 output tokens, raw measured-throughput arithmetic gives:

| Reference rate | One GPU, one128-request method run | Ideal four-shard time, excluding startup/imbalance |
|---|---:|---:|
|167.83tok/s|12.4min|3.1min|
|114.93tok/s|18.1min|4.5min|
|59.41tok/s|35.0min|8.8min|
|26.05tok/s AR|79.9min|20.0min|

If every request reaches2048 output tokens, these token-dependent times grow by approximately2.10x. Actual lengths and rates vary by workload, target, fitted interface, and block size. Use summed request time for throughput and actual worker wall time for scheduling.

## Why the full matrix takes days

There are five trained interface variants in the two tracks together: ZIP-feature, direct token CE, direct AUF, fusion CE-LoRA, fusion AUF-LoRA. The frozen ZIP starting interface can be reused where its hashes/configuration match; it need not be counted as a sixth fit.

The four principal transfer pairs × four workloads × five trained variants × three fit seeds × three runtime repeats produce720 units of128 requests, or92,160 generated answers. This count alone excludes AR/native controls, scaling curves, development sweeps, EAGLE extensions, serving/profiling, full benchmark breadth, and Transformers replication. Runtime repetition need not rerun deterministic quality scoring on identical arrays, but it still measures generation again.

At the observed Qwen output volume and167.83tok/s, those720 units alone cost about149 GPU-hours; at59.41tok/s, about420 GPU-hours. These are sensitivity endpoints, not assumed uniform rates. AR controls are shared across fitting seeds when their target/runtime/request hashes match, but their three timing repetitions remain real runs. Scaling trajectories save fitting restarts; their128-request decoding endpoints still cost inference time.

## Provisional resource allowance

This allowance estimates the entire matrix, including the conditional family/secondary-architecture work; it is not a measured or exactly enumerated launch manifest. Deduplication and the actual supported cells will revise it.

| Work | Planning allowance in GPU-hours | Main uncertainty |
|---|---:|---|
| Batched rollouts, dense-prefix capture, source feature targets |40–100|Target changes, usable caches, full-prefix I/O |
| All loss/epoch/data/capacity fits |100–300|AUF backward cost, prefix lengths, microbatch throughput |
| Development, main, seed/repeat, breadth and backend evaluations |450–850|Output lengths, poor-acceptance cells, supported matrix size |
| Profiling and targeted reruns |20–50|Backend compatibility and diagnosis |
| **Total** |**610–1,300**|Must be recalibrated after the pilot |

Dividing by four gives about153–325 hours (6.4–13.5 days) of ideal fully occupied GPU time. That division does not account for serial engineering, load imbalance, compilation, host/storage contention, and paper preparation. Some CPU analysis can overlap GPU work; isolated timing cannot overlap training on the same device.

The AUF fit allowance is deliberately provisional: the historical14.5-minute ZIP fit only trained a feature mapper against cached vectors. AUF needs autograd through frozen DFlash operations and the vocabulary head. Likewise, quarter-density ZIP features do not replace the full conditioning prefixes needed by the planned block training.

## Cumulative milestones

| Milestone | Estimated elapsed time from implementation start |
|---|---:|
| First Qwen8B AUF versus CE/ZIP pilot, one128-request workload,2048 cap |6–12hours |
| Initial principal transfer results at one seed/repeat |1–2days, excluding unresolved cross-family cells |
| Main transfer comparisons across workloads with seeds/repeats |3–6days |
| Full study with the shared-vocabulary fallback, profiling, final Transformers checks and manuscript |8–16days |
| Full study including a successfully validated cross-family vLLM/AUF extension |Budget10–18days |

These milestones are cumulative, not durations to add together. First pilot and initial transfer results are useful measurements but do not satisfy the final seed/repeat/breadth requirements. Cross-family can fail feasibility; the user-approved fallback avoids indefinite architecture work. No fixed time can guarantee a successful heterogeneous-vocabulary implementation. A genuine integration redesign or scratch-capacity problem can exceed these estimates; report the specific blocker and scope decision rather than letting an unbounded search run.

Planning allocations for engineering, which can partly overlap GPU work:6–12hours for the initial trainer/export/runtime pilot, a bounded cross-family investigation alongside the core path, and8–16hours for final analysis/manuscript/PDF audit once evidence is ready. These are allowances rather than historical observations.

## Re-estimation procedure

After the first two hours of actual implementation/allocated pilot work, report what is working and remaining blockers. After the first successful fit/evaluation, replace guesses using:

1. Captured bytes/record and measured capture records/s for actual required prefixes.
2. AUF/CE updates/s and blocks/s in representative prefix-length buckets.
3. Actual token totals and request latency for AR and each trained method.
4. A deduplicated generated list of fit trajectories and128-request evaluation units.
5. Measured GPU occupancy, job setup time, and observed usable cache coverage.

Recompute remaining GPU-hours and the dependency critical path. Do not silently reduce128 requests,2048 output cap, fit seeds, or required controls to match the initial estimate. Any scope reduction must be explicit. The core result set can be delivered earlier while the rest of the agreed matrix continues.
