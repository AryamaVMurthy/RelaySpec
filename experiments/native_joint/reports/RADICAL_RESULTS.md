# Rapid decoding hypothesis screens

Target: at least10% end-to-end throughput improvement over original DFlash. These adaptive short screens do not establish a win: larger paired evaluation and output-quality checks are required. Intervals are paired request bootstraps and do not include search selection uncertainty. Numerical output identity with original block16 decoding is reported explicitly.

| Run/lane | Hypothesis | Candidate TPS | Original TPS | Ratio [95% interval] | Exact outputs | Draft calls skipped | Seconds for test |
|---|---|---:|---:|---|---:|---:|---:|
| run-29350/lane2 | ddtree_adapted_budget47 | 152.4 | 118.8 | 1.283 [1.134, 1.375] | 4/8 | 0.0% | 77.8 |
| run-29356/lane2 | ddtree_budget63_pilot | 156.0 | 121.7 | 1.282 [1.112, 1.383] | 4/8 | 0.0% | 76.0 |
| run-29363/lane3 | ddtree47_temperature0.75 | 153.5 | 120.1 | 1.278 [1.111, 1.369] | 3/8 | 0.0% | 39.0 |
| run-29356/lane2 | ddtree_budget95_pilot | 154.3 | 121.7 | 1.268 [1.063, 1.376] | 3/8 | 0.0% | 76.1 |
| run-29350/lane1 | ddtree_adapted_budget31 | 150.3 | 121.7 | 1.235 [1.071, 1.320] | 3/8 | 0.0% | 76.2 |
| run-29363/lane3 | ddtree47_temperature1.25 | 147.6 | 120.4 | 1.226 [1.060, 1.314] | 3/8 | 0.0% | 39.7 |
| run-29356/lane1 | ddtree_budget47_16requests_cap2048 | 150.4 | 125.1 | 1.202 [1.095, 1.275] | 5/16 | 0.0% | 165.3 |
| run-29363/lane0 | compact_two_tap_5layer_ddtree47 | 144.6 | 121.3 | 1.192 [1.014, 1.296] | 3/8 | 0.0% | 112.0 |
| run-29356/lane0 | ddtree_budget31_16requests_cap2048 | 146.9 | 124.9 | 1.176 [1.080, 1.261] | 5/16 | 0.0% | 164.4 |
| run-29363/lane3 | ddtree47_temperature0.5 | 140.0 | 119.7 | 1.169 [1.050, 1.227] | 3/8 | 0.0% | 40.6 |
| run-29356/lane3 | gpu_leaf16_prefix0.5 | 141.6 | 121.8 | 1.162 [1.057, 1.226] | 3/8 | 0.0% | 39.9 |
| run-29356/lane3 | gpu_leaf16_prefix1.0 | 141.3 | 121.7 | 1.161 [1.060, 1.221] | 3/8 | 0.0% | 40.0 |
| run-29356/lane3 | gpu_leaf16_prefix0.0 | 141.1 | 121.8 | 1.159 [1.043, 1.219] | 3/8 | 0.0% | 40.4 |
| run-29337/lane3 | leaf_top5_p4_main16 | 140.9 | 121.8 | 1.158 [1.025, 1.228] | 4/8 | 0.0% | 40.3 |
| run-29350/lane0 | ddtree_adapted_budget15 | 139.4 | 121.7 | 1.145 [1.015, 1.225] | 6/8 | 0.0% | 80.1 |
| run-29337/lane3 | leaf_top4_p5_main16 | 138.8 | 121.7 | 1.140 [1.033, 1.200] | 3/8 | 0.0% | 40.0 |
| run-29363/lane3 | ddtree47_temperature1.5 | 136.7 | 120.1 | 1.138 [1.015, 1.213] | 4/8 | 0.0% | 40.3 |
| run-29337/lane2 | budget_tree5_main16_alt6 | 138.1 | 121.6 | 1.135 [1.033, 1.186] | 3/8 | 0.0% | 40.3 |
| run-29338/lane2 | leaf_top5_p4_strict_bf16 | 138.5 | 122.1 | 1.135 [0.997, 1.222] | 3/8 | 0.0% | 80.5 |
| run-29337/lane3 | leaf_top4_p4_main16 | 137.2 | 121.7 | 1.127 [1.026, 1.184] | 4/8 | 0.0% | 40.3 |
| run-29337/lane3 | leaf_top4_p6_main14 | 136.5 | 121.8 | 1.121 [0.988, 1.191] | 4/8 | 0.0% | 40.9 |
| run-29338/lane0 | leaf_top5_p4_breadth | 144.0 | 128.6 | 1.120 [1.065, 1.168] | 14/32 | 0.0% | 307.3 |
| run-29338/lane3 | leaf_top8_p3_budget | 136.2 | 121.8 | 1.118 [0.975, 1.211] | 3/8 | 0.0% | 41.0 |
| run-29338/lane3 | leaf_top6_p4_budget | 135.6 | 121.8 | 1.113 [0.991, 1.177] | 4/8 | 0.0% | 41.1 |
| run-29338/lane3 | leaf_top8_p4_budget | 135.1 | 121.9 | 1.108 [0.977, 1.178] | 4/8 | 0.0% | 41.1 |
| run-29338/lane2 | tree5_suffix6_strict_bf16 | 135.3 | 122.1 | 1.108 [0.990, 1.177] | 3/8 | 0.0% | 82.4 |
| run-29338/lane3 | leaf_top6_p3_budget | 135.0 | 121.8 | 1.108 [0.997, 1.171] | 3/8 | 0.0% | 41.2 |
| run-29337/lane2 | budget_tree4_main14_alt8 | 134.3 | 121.6 | 1.104 [0.985, 1.172] | 3/8 | 0.0% | 41.2 |
| run-29337/lane2 | budget_tree3_main16_alt8 | 133.6 | 121.2 | 1.103 [1.004, 1.154] | 4/8 | 0.0% | 41.5 |
| run-29315/lane3 | tree4_suffix8 | 133.8 | 121.7 | 1.099 [0.979, 1.163] | 3/8 | 0.0% | 41.3 |
| run-29315/lane3 | tree8_suffix8 | 133.8 | 121.7 | 1.099 [0.993, 1.154] | 4/8 | 0.0% | 40.7 |
| run-29338/lane1 | tree5_suffix6_breadth | 141.1 | 128.9 | 1.095 [1.057, 1.132] | 10/32 | 0.0% | 309.3 |
| run-29337/lane2 | budget_tree4_main12_alt8 | 133.1 | 121.6 | 1.094 [0.928, 1.185] | 3/8 | 0.0% | 41.4 |
| run-29315/lane2 | leaf_top3_p8 | 132.7 | 121.4 | 1.093 [1.015, 1.136] | 3/8 | 0.0% | 40.6 |
| run-29315/lane3 | tree4_suffix12 | 133.0 | 121.8 | 1.092 [0.995, 1.147] | 3/8 | 0.0% | 40.6 |
| run-29337/lane2 | budget_tree8_main16_alt4 | 132.9 | 121.6 | 1.092 [1.019, 1.127] | 3/8 | 0.0% | 40.7 |
| run-29315/lane2 | leaf_top3_p4 | 132.4 | 121.4 | 1.091 [1.014, 1.133] | 3/8 | 0.0% | 40.8 |
| run-29307/lane1 | tree4_b16_first_mNone | 131.4 | 120.7 | 1.088 [0.975, 1.144] | 3/8 | 0.0% | 41.9 |
| run-29315/lane1 | leaf_top2_p12 | 132.1 | 121.9 | 1.084 [0.997, 1.128] | 3/8 | 0.0% | 41.6 |
| run-29315/lane1 | leaf_top2_p8 | 131.7 | 121.9 | 1.080 [1.008, 1.116] | 3/8 | 0.0% | 41.4 |
| run-29315/lane1 | leaf_top2_p4 | 131.5 | 121.9 | 1.079 [1.002, 1.115] | 4/8 | 0.0% | 41.4 |
| run-29315/lane1 | leaf_top2_p15 | 131.5 | 121.9 | 1.079 [0.996, 1.121] | 3/8 | 0.0% | 41.7 |
| run-29350/lane3 | leaf_top5_p4_2048_pilot | 124.9 | 116.0 | 1.076 [1.013, 1.143] | 4/8 | 0.0% | 54.9 |
| run-29315/lane2 | leaf_top3_p15 | 130.6 | 121.4 | 1.076 [0.981, 1.123] | 3/8 | 0.0% | 41.9 |
| run-29307/lane2 | tree4_b16_first_m2.0 | 130.1 | 121.5 | 1.071 [0.996, 1.114] | 3/8 | 0.0% | 41.3 |
| run-29307/lane0 | tree2_b16_weakest_mNone | 129.8 | 121.7 | 1.067 [1.008, 1.097] | 3/8 | 0.0% | 41.5 |
| run-29307/lane2 | tree4_b16_first_m1.0 | 128.3 | 121.5 | 1.056 [0.977, 1.104] | 5/8 | 0.0% | 41.9 |
| run-29315/lane0 | tree4_b16_breadth | 135.9 | 128.7 | 1.056 [1.015, 1.095] | 10/32 | 0.0% | 317.6 |
| run-29315/lane2 | leaf_top3_p12 | 127.8 | 121.4 | 1.053 [0.978, 1.097] | 4/8 | 0.0% | 41.5 |
| run-29337/lane1 | tree4_suffix8_breadth | 135.2 | 128.5 | 1.052 [1.012, 1.088] | 12/32 | 0.0% | 315.8 |
| run-29307/lane0 | tree2_b16_first_mNone | 127.6 | 121.7 | 1.049 [0.971, 1.100] | 4/8 | 0.0% | 42.1 |
| run-29184/lane2 | lookup_s3_b32 | 139.1 | 134.1 | 1.037 [0.956, 1.178] | 5/8 | 10.4% | 25.3 |
| run-29315/lane3 | tree8_suffix12 | 126.0 | 121.7 | 1.035 [0.946, 1.081] | 3/8 | 0.0% | 42.6 |
| run-29363/lane1 | compact_four_drop3_ddtree47 | 124.4 | 121.7 | 1.023 [0.826, 1.164] | 3/8 | 0.0% | 119.6 |
| run-29307/lane1 | tree8_b16_first_mNone | 123.2 | 120.9 | 1.019 [0.958, 1.061] | 3/8 | 0.0% | 42.8 |
| run-29307/lane2 | tree4_b16_first_m0.5 | 123.2 | 121.5 | 1.014 [0.982, 1.040] | 3/8 | 0.0% | 42.8 |
| run-29363/lane2 | compact_four_drop2_ddtree47 | 123.3 | 121.6 | 1.014 [0.833, 1.157] | 3/8 | 0.0% | 119.3 |
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
| run-29307/lane0 | tree1_b16_first_mNone | 120.0 | 121.6 | 0.987 [0.986, 0.987] | 8/8 | 0.0% | 42.6 |
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
| run-29307/lane3 | tree4_b24_first_mNone | 113.6 | 121.6 | 0.934 [0.868, 0.969] | 3/8 | 0.0% | 44.9 |
| run-29288/lane3 | lazy_head_b32_c16 | 113.4 | 121.8 | 0.931 [0.873, 0.963] | 3/8 | 0.0% | 44.2 |
| run-29184/lane1 | block12 | 124.7 | 134.0 | 0.930 [0.872, 0.975] | 4/8 | 0.0% | 26.8 |
| run-29307/lane3 | tree4_b8_first_mNone | 112.6 | 121.5 | 0.927 [0.767, 1.018] | 3/8 | 0.0% | 45.2 |
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
| run-29301/lane2 | compiled_bf16_draft_and_head | 108.7 | 121.0 | 0.899 [0.891, 0.904] | 8/8 | 0.0% | 89.4 |
| run-29184/lane3 | adaptive_headroom2 | 120.0 | 134.0 | 0.896 [0.835, 0.935] | 3/8 | 0.0% | 27.3 |
| run-29301/lane0 | compiled_bf16_draft_only | 107.6 | 120.8 | 0.890 [0.875, 0.895] | 8/8 | 0.0% | 90.0 |
| run-29285/lane1 | target_warm_0.5 | 107.2 | 121.5 | 0.882 [0.816, 0.914] | 8/8 | 0.0% | 45.2 |
| run-29301/lane3 | compiled_int4_draft_and_head | 106.0 | 121.0 | 0.876 [0.856, 0.899] | 8/8 | 0.0% | 90.6 |
| run-29285/lane2 | target_tail_min10_rounds1 | 104.0 | 121.7 | 0.855 [0.794, 0.887] | 6/8 | 41.9% | 45.9 |
| run-29288/lane0 | lazy_head_b16_c1 | 103.8 | 121.5 | 0.854 [0.755, 0.919] | 8/8 | 0.0% | 46.0 |
| run-29192/lane3 | recycle_min8_rounds1 | 114.6 | 134.3 | 0.853 [0.775, 0.920] | 6/8 | 41.3% | 27.9 |
| run-29184/lane1 | block8 | 113.8 | 133.9 | 0.850 [0.744, 0.939] | 4/8 | 0.0% | 28.0 |
| run-29192/lane3 | recycle_min4_rounds1 | 112.8 | 134.3 | 0.840 [0.762, 0.900] | 5/8 | 44.3% | 28.2 |
| run-29184/lane3 | branch_b8_first | 111.4 | 134.0 | 0.831 [0.727, 0.911] | 8/8 | 0.0% | 28.4 |
| run-29301/lane1 | compiled_int4_draft_only | 100.5 | 121.3 | 0.829 [0.709, 0.869] | 8/8 | 0.0% | 93.1 |
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

Lanes failing before hypothesis timing: 8. These are preparation/engineering failures, not measured regressions.
- run-29296/lane0: exit1; log `reports/run-29296/lane0.log`.
- run-29296/lane1: exit1; log `reports/run-29296/lane1.log`.
- run-29296/lane2: exit1; log `reports/run-29296/lane2.log`.
- run-29296/lane3: exit1; log `reports/run-29296/lane3.log`.
- run-29298/lane0: exit1; log `reports/run-29298/lane0.log`.
- run-29298/lane1: exit1; log `reports/run-29298/lane1.log`.
- run-29298/lane2: exit1; log `reports/run-29298/lane2.log`.
- run-29298/lane3: exit1; log `reports/run-29298/lane3.log`.
