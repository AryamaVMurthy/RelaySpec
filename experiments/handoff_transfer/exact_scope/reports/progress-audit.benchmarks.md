Experiment audit: **incomplete**.

**Provisional references: this table can compare different physical GPUs. Do not interpret small differences as isolated method gains. Use the GPU-matched report for final comparisons.**

Verified fits: 20/30. Fully evaluated trained cells: 20/30. Transformers confirmations: 0/2.

Only complete, audited three-repetition comparisons appear below. These are fixed development requests, one fitting seed, 128 requests per pass, and a 2048-token cap with natural EOS. Throughput is aggregate batched output tokens/s, not single-request latency. Ranges describe timing repetitions, not fitting-seed uncertainty. Each speedup is the mean of ratios paired by timing repetition and model family. Reference passes are reused across jobs on the same node and GPU model; comparisons can involve different physical GPUs. The audit retains benchmark job IDs and GPU UUIDs. Accepted/draft is the measured accepted draft-token count divided by the draft-block count; it excludes the verifier bonus token.

| Family | Method | Batch | Mean TPS | TPS range | / AR | / Original | / ZIP | Accepted/draft |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama | llama/dense_fusion/auf | 128 | 4651.7 | 4650.1–4654.3 | 2.937 | 1.134 | 1.117 | 4.441 |
| llama | llama/dense_fusion/ce | 128 | 4585.2 | 4575.6–4596.7 | 2.895 | 1.118 | 1.101 | 4.346 |
| llama | llama/feature/feature_ce | 128 | 1104.9 | 1099.0–1116.3 | 0.698 | 0.269 | 0.265 | 0.196 |
| llama | llama/feature/forward_kl | 128 | 1108.7 | 1105.7–1112.1 | 0.700 | 0.270 | 0.266 | 0.196 |
| llama | llama/feature/reverse_kl | 128 | 1123.2 | 1120.9–1124.5 | 0.709 | 0.274 | 0.270 | 0.176 |
| llama | llama/five_ba56/auf | 128 | 4579.2 | 4568.7–4584.9 | 2.891 | 1.117 | 1.099 | 4.357 |
| llama | llama/five_ba56/ce | 128 | 4479.7 | 4479.2–4480.4 | 2.828 | 1.092 | 1.076 | 4.253 |
| llama | llama/five_maps/auf | 128 | 4672.5 | 4663.6–4687.0 | 2.950 | 1.139 | 1.122 | 4.398 |
| llama | llama/five_maps/ce | 128 | 4546.2 | 4531.5–4562.1 | 2.870 | 1.109 | 1.091 | 4.308 |
| llama | llama/normal_ce/ce | 128 | 4541.3 | 4533.6–4549.0 | 2.867 | 1.107 | 1.090 | 4.332 |
| llama | original | 128 | 4100.8 | 4090.8–4115.6 | 2.589 | 1.000 | 0.985 | 3.899 |
| llama | zip | 128 | 4165.1 | 4150.3–4172.7 | 2.630 | 1.016 | 1.000 | 3.883 |
| q8 | original | 128 | 2696.2 | 2694.1–2698.0 | 2.193 | 1.000 | 1.002 | 6.192 |
| q8 | q8/dense_fusion/auf | 128 | 2844.8 | 2831.8–2856.2 | 2.314 | 1.055 | 1.057 | 6.471 |
| q8 | q8/dense_fusion/ce | 128 | 2842.6 | 2828.8–2860.1 | 2.312 | 1.054 | 1.056 | 6.324 |
| q8 | q8/feature/feature_ce | 128 | 2258.4 | 2243.8–2266.5 | 1.837 | 0.838 | 0.839 | 5.002 |
| q8 | q8/feature/forward_kl | 128 | 2185.7 | 2171.8–2201.5 | 1.778 | 0.811 | 0.812 | 5.002 |
| q8 | q8/feature/reverse_kl | 128 | 1838.7 | 1837.6–1840.1 | 1.496 | 0.682 | 0.683 | 3.869 |
| q8 | q8/five_ba56/auf | 128 | 2822.1 | 2808.0–2837.3 | 2.296 | 1.047 | 1.048 | 6.433 |
| q8 | q8/five_ba56/ce | 128 | 2820.0 | 2818.2–2821.2 | 2.294 | 1.046 | 1.048 | 6.303 |
| q8 | q8/five_maps/auf | 128 | 2788.6 | 2786.1–2792.7 | 2.269 | 1.034 | 1.036 | 6.100 |
| q8 | q8/five_maps/ce | 128 | 2685.7 | 2681.7–2688.8 | 2.185 | 0.996 | 0.998 | 5.976 |
| q8 | q8/normal_ce/ce | 128 | 2758.2 | 2746.9–2771.8 | 2.244 | 1.023 | 1.025 | 6.241 |
| q8 | zip | 128 | 2692.0 | 2674.4–2703.2 | 2.190 | 0.998 | 1.000 | 6.224 |
