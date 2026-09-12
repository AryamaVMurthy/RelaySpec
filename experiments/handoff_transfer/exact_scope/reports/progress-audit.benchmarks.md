Experiment audit: **incomplete**.

Verified fits: 20/30. Fully evaluated trained cells: 8/30. Transformers confirmations: 0/2.

Only complete, audited three-repetition comparisons appear below. These are fixed development requests, one fitting seed, 128 requests per pass, and a 2048-token cap with natural EOS. Throughput is aggregate batched output tokens/s, not single-request latency. Ranges describe timing repetitions, not fitting-seed uncertainty. Each speedup is the mean of ratios paired by timing repetition and model family. Reference passes are reused across jobs on the same node and GPU model; comparisons can involve different physical GPUs. The audit retains benchmark job IDs and GPU UUIDs. Accepted/draft is the measured accepted draft-token count divided by the draft-block count; it excludes the verifier bonus token.

| Family | Method | Batch | Mean TPS | TPS range | / AR | / Original | / ZIP | Accepted/draft |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| q8 | original | 128 | 2696.2 | 2694.1–2698.0 | 2.193 | 1.000 | 1.002 | 6.192 |
| q8 | q8/dense_fusion/auf | 128 | 2844.8 | 2831.8–2856.2 | 2.314 | 1.055 | 1.057 | 6.471 |
| q8 | q8/dense_fusion/ce | 128 | 2842.6 | 2828.8–2860.1 | 2.312 | 1.054 | 1.056 | 6.324 |
| q8 | q8/feature/feature_ce | 128 | 2258.4 | 2243.8–2266.5 | 1.837 | 0.838 | 0.839 | 5.002 |
| q8 | q8/five_ba56/auf | 128 | 2822.1 | 2808.0–2837.3 | 2.296 | 1.047 | 1.048 | 6.433 |
| q8 | q8/five_ba56/ce | 128 | 2820.0 | 2818.2–2821.2 | 2.294 | 1.046 | 1.048 | 6.303 |
| q8 | q8/five_maps/auf | 128 | 2788.6 | 2786.1–2792.7 | 2.269 | 1.034 | 1.036 | 6.100 |
| q8 | q8/five_maps/ce | 128 | 2685.7 | 2681.7–2688.8 | 2.185 | 0.996 | 0.998 | 5.976 |
| q8 | q8/normal_ce/ce | 128 | 2758.2 | 2746.9–2771.8 | 2.244 | 1.023 | 1.025 | 6.241 |
| q8 | zip | 128 | 2692.0 | 2674.4–2703.2 | 2.190 | 0.998 | 1.000 | 6.224 |
