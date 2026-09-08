# Paired standalone evaluation

Intervals resample paired requests with all timing repeats kept together. They exclude fitting-seed and adaptive-selection uncertainty. Throughput includes prefill. Exact outputs compare token arrays with native DFlash; AR differences, if present, are recorded separately and are not a quality score.

| Run/lane | Workload | Method | Block | Requests × repeats | TPS | Native TPS | Ratio [95% interval] | Exact native requests |
|---|---|---|---:|---:|---:|---:|---|---:|
| run-29166/lane0 | gsm8k | student | 16 | 8 × 2 | 185.3 | 184.6 | 1.004 [0.958, 1.054] | 8/8 |
| run-29166/lane0 | gsm8k | ar | 1 | 8 × 2 | 38.1 | 184.6 | 0.207 [0.184, 0.234] | 4/8 |
| run-29166/lane1 | math500 | student | 16 | 8 × 2 | 200.6 | 209.2 | 0.959 [0.929, 0.980] | 8/8 |
| run-29166/lane1 | math500 | ar | 1 | 8 × 2 | 37.7 | 209.2 | 0.180 [0.147, 0.214] | 3/8 |
| run-29166/lane2 | humaneval | student | 16 | 8 × 2 | 154.9 | 171.8 | 0.902 [0.873, 0.931] | 8/8 |
| run-29166/lane2 | humaneval | ar | 1 | 8 × 2 | 38.2 | 171.8 | 0.222 [0.204, 0.246] | 1/8 |
| run-29166/lane3 | mtbench | student | 16 | 8 × 2 | 57.1 | 58.8 | 0.971 [0.957, 0.989] | 8/8 |
| run-29166/lane3 | mtbench | ar | 1 | 8 × 2 | 37.8 | 58.8 | 0.643 [0.579, 0.692] | 1/8 |
