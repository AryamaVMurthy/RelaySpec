# Rapid decoding hypothesis screens

Target: at least10% end-to-end throughput improvement over original DFlash. These adaptive short screens do not establish a win: larger paired evaluation and output-quality checks are required. Intervals are paired request bootstraps and do not include search selection uncertainty. Numerical output identity with original block16 decoding is reported explicitly.

| Run/lane | Hypothesis | Candidate TPS | Original TPS | Ratio [95% interval] | Exact outputs | Draft calls skipped | Seconds for test |
|---|---|---:|---:|---|---:|---:|---:|
| run-29184/lane2 | lookup_s3_b32 | 139.1 | 134.1 | 1.037 [0.956, 1.178] | 5/8 | 10.4% | 25.3 |
| run-29192/lane1 | lookup_s4_b32 | 132.4 | 131.0 | 1.010 [0.976, 1.066] | 12/16 | 7.7% | 56.8 |
| run-29184/lane0 | confidence0.3 | 134.8 | 133.7 | 1.008 [0.948, 1.090] | 4/8 | 0.0% | 25.7 |
| run-29184/lane3 | branch_b16_first | 134.4 | 134.0 | 1.003 [0.971, 1.023] | 3/8 | 0.0% | 25.7 |
| run-29285/lane0 | target_warm_0.2 | 121.7 | 121.7 | 1.000 [0.992, 1.007] | 8/8 | 0.0% | 42.3 |
| run-29192/lane0 | lookup_s3_b32_code8 | 171.7 | 171.7 | 1.000 [0.947, 1.058] | 4/8 | 16.1% | 72.9 |
| run-29288/lane0 | lazy_head_b16_c16 | 121.4 | 121.5 | 0.999 [0.998, 1.001] | 8/8 | 0.0% | 42.4 |
| run-29192/lane0 | confidence03_code8 | 171.5 | 171.6 | 0.999 [0.945, 1.071] | 1/8 | 0.0% | 74.1 |
| run-29282/lane2 | warm_0.0 | 121.5 | 121.6 | 0.999 [0.999, 0.999] | 8/8 | 0.0% | 42.4 |
| run-29203/lane0 | vocab_top1024 | 121.4 | 121.6 | 0.998 [0.993, 1.009] | 8/8 | 0.0% | 42.4 |
| run-29285/lane0 | target_warm_0.1 | 121.3 | 121.7 | 0.996 [0.988, 1.006] | 8/8 | 0.0% | 42.4 |
| run-29282/lane2 | warm_0.05 | 121.0 | 121.6 | 0.995 [0.991, 1.006] | 8/8 | 0.0% | 42.4 |
| run-29192/lane1 | lookup_s3_b48 | 130.3 | 131.1 | 0.994 [0.952, 1.045] | 11/16 | 10.2% | 57.3 |
| run-29192/lane2 | conditional_branch_1.0 | 130.1 | 130.9 | 0.994 [0.932, 1.060] | 4/8 | 0.0% | 26.5 |
| run-29285/lane0 | target_warm_0.05 | 120.9 | 121.7 | 0.993 [0.987, 1.003] | 8/8 | 0.0% | 42.4 |
| run-29282/lane2 | warm_0.2 | 120.6 | 121.6 | 0.991 [0.968, 1.012] | 8/8 | 0.0% | 42.5 |
| run-29288/lane2 | lazy_head_b16_c8 | 120.6 | 121.7 | 0.991 [0.980, 0.997] | 8/8 | 0.0% | 42.5 |
| run-29282/lane2 | warm_0.1 | 120.4 | 121.6 | 0.990 [0.984, 1.002] | 8/8 | 0.0% | 42.5 |
| run-29184/lane0 | verify12 | 132.3 | 133.7 | 0.990 [0.937, 1.028] | 4/8 | 0.0% | 25.9 |
| run-29184/lane2 | lookup_s4_b16 | 132.3 | 134.1 | 0.986 [0.960, 1.009] | 8/8 | 9.4% | 25.9 |
| run-29203/lane3 | long_vocab_top512 | 114.0 | 115.8 | 0.984 [0.968, 0.995] | 8/8 | 0.0% | 54.8 |
| run-29192/lane2 | conditional_branch_0.5 | 128.0 | 130.8 | 0.979 [0.950, 0.998] | 4/8 | 0.0% | 26.7 |
| run-29282/lane3 | warm_0.35 | 116.8 | 119.5 | 0.977 [0.943, 0.996] | 8/8 | 0.0% | 43.6 |
| run-29192/lane2 | four_branches | 128.7 | 131.7 | 0.977 [0.921, 1.010] | 3/8 | 0.0% | 26.5 |
| run-29192/lane2 | conditional_branch_2.0 | 127.5 | 130.9 | 0.974 [0.953, 0.991] | 5/8 | 0.0% | 26.7 |
| run-29288/lane1 | lazy_head_b16_c4 | 118.3 | 121.6 | 0.973 [0.942, 0.990] | 8/8 | 0.0% | 42.9 |
| run-29184/lane2 | lookup_s3_b16 | 130.4 | 134.1 | 0.972 [0.930, 1.002] | 8/8 | 10.9% | 26.1 |
| run-29184/lane3 | branch_b16_weakest | 130.0 | 134.0 | 0.970 [0.933, 1.004] | 3/8 | 0.0% | 26.1 |
| run-29203/lane0 | vocab_top256 | 117.9 | 121.7 | 0.969 [0.954, 0.992] | 8/8 | 0.0% | 43.0 |
| run-29285/lane1 | target_warm_0.35 | 117.5 | 121.6 | 0.966 [0.942, 0.989] | 8/8 | 0.0% | 43.1 |
| run-29297/lane3 | int4_g128_draft_and_head | 117.1 | 121.5 | 0.964 [0.941, 0.990] | 8/8 | 0.0% | 85.9 |
| run-29297/lane2 | int4_g128_draft_only | 115.8 | 121.6 | 0.953 [0.932, 0.963] | 8/8 | 0.0% | 86.3 |
| run-29297/lane0 | int4_g32_draft_only | 114.4 | 120.7 | 0.947 [0.934, 0.958] | 8/8 | 0.0% | 87.2 |
| run-29192/lane1 | lookup_s2_b32 | 124.0 | 131.0 | 0.947 [0.885, 0.992] | 10/16 | 17.1% | 58.7 |
| run-29297/lane1 | int4_g32_draft_and_head | 114.4 | 121.3 | 0.943 [0.924, 0.954] | 8/8 | 0.0% | 87.0 |
| run-29184/lane0 | confidence0.6 | 125.1 | 133.7 | 0.935 [0.859, 0.999] | 5/8 | 0.0% | 26.7 |
| run-29288/lane3 | lazy_head_b32_c16 | 113.4 | 121.8 | 0.931 [0.873, 0.963] | 3/8 | 0.0% | 44.2 |
| run-29184/lane1 | block12 | 124.7 | 134.0 | 0.930 [0.872, 0.975] | 4/8 | 0.0% | 26.8 |
| run-29184/lane1 | block24 | 124.0 | 134.0 | 0.925 [0.872, 0.957] | 3/8 | 0.0% | 26.8 |
| run-29288/lane3 | lazy_head_b32_c8 | 112.7 | 121.8 | 0.925 [0.862, 0.959] | 3/8 | 0.0% | 44.4 |
| run-29184/lane2 | lookup_s2_b16 | 123.4 | 134.1 | 0.921 [0.838, 0.974] | 8/8 | 18.8% | 26.8 |
| run-29203/lane3 | long_vocab_top64 | 106.5 | 115.8 | 0.920 [0.868, 0.946] | 8/8 | 0.0% | 56.7 |
| run-29288/lane1 | lazy_head_b16_c2 | 111.9 | 121.6 | 0.920 [0.830, 0.966] | 8/8 | 0.0% | 44.2 |
| run-29288/lane2 | lazy_head_b32_c4 | 110.7 | 121.7 | 0.909 [0.834, 0.949] | 3/8 | 0.0% | 44.8 |
| run-29203/lane0 | vocab_top64 | 110.0 | 121.6 | 0.905 [0.867, 0.932] | 8/8 | 0.0% | 44.5 |
| run-29282/lane3 | warm_0.5 | 109.6 | 121.3 | 0.904 [0.856, 0.927] | 8/8 | 0.0% | 44.7 |
| run-29184/lane0 | verify8 | 120.3 | 133.7 | 0.900 [0.785, 0.994] | 4/8 | 0.0% | 27.2 |
| run-29184/lane1 | block32 | 120.6 | 134.0 | 0.900 [0.848, 0.931] | 3/8 | 0.0% | 27.2 |
| run-29184/lane3 | adaptive_headroom2 | 120.0 | 134.0 | 0.896 [0.835, 0.935] | 3/8 | 0.0% | 27.3 |
| run-29285/lane1 | target_warm_0.5 | 107.2 | 121.5 | 0.882 [0.816, 0.914] | 8/8 | 0.0% | 45.2 |
| run-29285/lane2 | target_tail_min10_rounds1 | 104.0 | 121.7 | 0.855 [0.794, 0.887] | 6/8 | 41.9% | 45.9 |
| run-29288/lane0 | lazy_head_b16_c1 | 103.8 | 121.5 | 0.854 [0.755, 0.919] | 8/8 | 0.0% | 46.0 |
| run-29192/lane3 | recycle_min8_rounds1 | 114.6 | 134.3 | 0.853 [0.775, 0.920] | 6/8 | 41.3% | 27.9 |
| run-29184/lane1 | block8 | 113.8 | 133.9 | 0.850 [0.744, 0.939] | 4/8 | 0.0% | 28.0 |
| run-29192/lane3 | recycle_min4_rounds1 | 112.8 | 134.3 | 0.840 [0.762, 0.900] | 5/8 | 44.3% | 28.2 |
| run-29184/lane3 | branch_b8_first | 111.4 | 134.0 | 0.831 [0.727, 0.911] | 8/8 | 0.0% | 28.4 |
| run-29285/lane2 | target_tail_min6_rounds1 | 100.8 | 121.7 | 0.829 [0.747, 0.862] | 6/8 | 44.8% | 46.7 |
| run-29285/lane2 | target_tail_min3_rounds1 | 100.3 | 121.7 | 0.825 [0.743, 0.858] | 5/8 | 45.9% | 46.8 |
| run-29203/lane0 | vocab_top16 | 100.0 | 121.6 | 0.822 [0.751, 0.870] | 8/8 | 0.0% | 47.0 |
| run-29203/lane2 | long_context_w128 | 95.2 | 115.9 | 0.821 [0.651, 0.908] | 8/8 | 0.0% | 60.2 |
| run-29203/lane1 | context_w128 | 98.4 | 121.4 | 0.811 [0.646, 0.922] | 8/8 | 0.0% | 47.3 |
| run-29184/lane3 | branch_b8_weakest | 108.5 | 134.0 | 0.810 [0.713, 0.877] | 8/8 | 0.0% | 28.8 |
| run-29192/lane3 | recycle_min8_rounds2 | 104.1 | 134.3 | 0.776 [0.689, 0.845] | 6/8 | 56.1% | 29.4 |
| run-29285/lane3 | target_tail_min10_rounds2 | 94.0 | 121.5 | 0.774 [0.673, 0.823] | 5/8 | 57.1% | 48.5 |
| run-29192/lane3 | recycle_min4_rounds2 | 100.6 | 134.3 | 0.749 [0.659, 0.810] | 4/8 | 59.8% | 30.0 |
| run-29285/lane3 | target_tail_min6_rounds2 | 89.4 | 121.5 | 0.735 [0.628, 0.804] | 4/8 | 60.8% | 49.9 |
| run-29285/lane3 | target_tail_min3_rounds2 | 88.1 | 121.6 | 0.725 [0.617, 0.801] | 6/8 | 62.4% | 50.6 |
| run-29203/lane1 | context_w64 | 87.8 | 121.5 | 0.723 [0.553, 0.837] | 8/8 | 0.0% | 50.5 |
| run-29285/lane1 | target_warm_0.75 | 85.3 | 121.6 | 0.702 [0.592, 0.784] | 8/8 | 0.0% | 51.3 |
| run-29282/lane3 | warm_0.75 | 84.1 | 121.4 | 0.693 [0.591, 0.758] | 8/8 | 0.0% | 51.8 |
| run-29203/lane2 | long_context_w32 | 78.6 | 115.8 | 0.679 [0.486, 0.791] | 8/8 | 0.0% | 67.1 |
| run-29184/lane2 | adaptive_progress | 90.9 | 134.1 | 0.678 [0.534, 0.827] | 4/8 | 0.0% | 31.8 |
| run-29203/lane1 | context_w32 | 79.4 | 121.5 | 0.654 [0.481, 0.783] | 8/8 | 0.0% | 53.5 |
| run-29184/lane0 | verify4 | 87.2 | 133.6 | 0.652 [0.484, 0.837] | 5/8 | 0.0% | 32.6 |
| run-29184/lane1 | block4 | 83.6 | 133.9 | 0.624 [0.486, 0.774] | 5/8 | 0.0% | 33.5 |
| run-29203/lane1 | context_w16 | 71.1 | 121.4 | 0.586 [0.416, 0.730] | 8/8 | 0.0% | 57.4 |
| run-29282/lane3 | warm_1.0 | 43.4 | 121.4 | 0.357 [0.253, 0.479] | 8/8 | 0.0% | 80.4 |

Selection decision: the run29184 two-code-request history-lookup pilot showed1.332×, but its run29192 follow-up on eight code requests with two timing repeats obtained1.000× [0.947,1.058]. That pilot gain did not replicate and is not a promotion candidate. Ranking below/above by an adaptive point estimate does not establish superiority.

Failed hypotheses: 0. Error traces are retained in radical-summary.json and the raw run folders.

Lanes failing before hypothesis timing: 4. These are preparation/engineering failures, not measured regressions.
- run-29296/lane0: exit1; log `reports/run-29296/lane0.log`.
- run-29296/lane1: exit1; log `reports/run-29296/lane1.log`.
- run-29296/lane2: exit1; log `reports/run-29296/lane2.log`.
- run-29296/lane3: exit1; log `reports/run-29296/lane3.log`.
