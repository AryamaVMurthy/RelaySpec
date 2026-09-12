Experiment audit: **incomplete**.

Verified fits: 20/30. Fully evaluated trained cells: 3/30. Transformers confirmations: 0/2.

Only complete, audited three-repetition comparisons appear below. These are fixed development requests, one fitting seed, 128 requests per pass, and a 2048-token cap with natural EOS. Throughput is aggregate batched output tokens/s, not single-request latency. Ranges describe timing repetitions, not fitting-seed uncertainty. Each speedup is the mean of ratios paired by timing repetition and model family. Reference passes are reused across jobs on the same node and GPU model; comparisons can involve different physical GPUs. The audit retains benchmark job IDs and GPU UUIDs.

| Family | Method | Batch | Mean TPS | TPS range | / AR | / Original | / ZIP |
|---|---|---:|---:|---:|---:|---:|---:|
| q8 | original | 128 | 2696.2 | 2694.1–2698.0 | 2.193 | 1.000 | 1.002 |
| q8 | q8/five_maps/auf | 128 | 2788.6 | 2786.1–2792.7 | 2.269 | 1.034 | 1.036 |
| q8 | q8/five_maps/ce | 128 | 2685.7 | 2681.7–2688.8 | 2.185 | 0.996 | 0.998 |
| q8 | q8/normal_ce/ce | 128 | 2758.2 | 2746.9–2771.8 | 2.244 | 1.023 | 1.025 |
| q8 | zip | 128 | 2692.0 | 2674.4–2703.2 | 2.190 | 0.998 | 1.000 |
