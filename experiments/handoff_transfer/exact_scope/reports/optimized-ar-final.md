Existing invariant speculative measurements divided by new O3, async, non-invariant AR TPS. Same128 prompts and physical GPUs, batch1 cap2048 BF16. Different runtime configurations; no speculative remeasurement.

If outputs differ, TPS ratios are per-token throughput comparisons, not identical-output latency speedups. Request-time ratios may additionally reflect changed output lengths. O3 is the highest provided optimization level, not a guarantee of globally optimal tuning.

| Target | Method | TPS | / optimized AR TPS | Request time ratio | Exact optimized AR |
|---|---|---:|---:|---:|---:|
| q8 | optimized_ar | 46.15 | 1.000 | 1.000 | 128/128 |
| q8 | previous_ar | 24.39 | 0.529 | 0.514 | 22/128 |
| q8 | native | 169.08 | 3.664 | 3.566 | 22/128 |
| q8 | five_maps-auf | 170.79 | 3.701 | 3.602 | 22/128 |
| q8 | five_maps-ce | 167.88 | 3.638 | 3.540 | 22/128 |
| q8 | dense_fusion-auf | 179.57 | 3.891 | 3.787 | 22/128 |
| q8 | dense_fusion-ce | 176.07 | 3.815 | 3.713 | 22/128 |
| q8 | original | 172.88 | 3.746 | 3.646 | 22/128 |
| cross | optimized_ar | 46.93 | 1.000 | 1.000 | 128/128 |
| cross | previous_ar | 24.54 | 0.523 | 0.546 | 36/128 |
| cross | native | 119.45 | 2.545 | 2.660 | 36/128 |
| cross | five_maps-auf | 143.56 | 3.059 | 3.197 | 36/128 |
| cross | five_maps-ce | 136.55 | 2.910 | 3.041 | 36/128 |
| cross | dense_fusion-auf | 123.42 | 2.630 | 2.749 | 36/128 |
| cross | dense_fusion-ce | 117.32 | 2.500 | 2.613 | 36/128 |
| cross | original | 107.67 | 2.294 | 2.398 | 36/128 |
