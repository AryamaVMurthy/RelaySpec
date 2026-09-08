# Rapid decoding hypothesis screens

Target: at least10% end-to-end throughput improvement over original DFlash. These adaptive short screens do not establish a win: larger paired evaluation and output-quality checks are required. Intervals are paired request bootstraps and do not include search selection uncertainty. Numerical output identity with original block16 decoding is reported explicitly.

| Run/lane | Hypothesis | Candidate TPS | Original TPS | Ratio [95% interval] | Exact outputs | Draft calls skipped | Seconds for test |
|---|---|---:|---:|---|---:|---:|---:|
| run-29184/lane2 | lookup_s3_b32 | 139.1 | 134.1 | 1.037 [0.956, 1.178] | 5/8 | 10.4% | 25.3 |
| run-29184/lane0 | confidence0.3 | 134.8 | 133.7 | 1.008 [0.948, 1.090] | 4/8 | 0.0% | 25.7 |
| run-29184/lane3 | branch_b16_first | 134.4 | 134.0 | 1.003 [0.971, 1.023] | 3/8 | 0.0% | 25.7 |
| run-29184/lane0 | verify12 | 132.3 | 133.7 | 0.990 [0.937, 1.028] | 4/8 | 0.0% | 25.9 |
| run-29184/lane2 | lookup_s4_b16 | 132.3 | 134.1 | 0.986 [0.960, 1.009] | 8/8 | 9.4% | 25.9 |
| run-29184/lane2 | lookup_s3_b16 | 130.4 | 134.1 | 0.972 [0.930, 1.002] | 8/8 | 10.9% | 26.1 |
| run-29184/lane3 | branch_b16_weakest | 130.0 | 134.0 | 0.970 [0.933, 1.004] | 3/8 | 0.0% | 26.1 |
| run-29184/lane0 | confidence0.6 | 125.1 | 133.7 | 0.935 [0.859, 0.999] | 5/8 | 0.0% | 26.7 |
| run-29184/lane1 | block12 | 124.7 | 134.0 | 0.930 [0.872, 0.975] | 4/8 | 0.0% | 26.8 |
| run-29184/lane1 | block24 | 124.0 | 134.0 | 0.925 [0.872, 0.957] | 3/8 | 0.0% | 26.8 |
| run-29184/lane2 | lookup_s2_b16 | 123.4 | 134.1 | 0.921 [0.838, 0.974] | 8/8 | 18.8% | 26.8 |
| run-29184/lane0 | verify8 | 120.3 | 133.7 | 0.900 [0.785, 0.994] | 4/8 | 0.0% | 27.2 |
| run-29184/lane1 | block32 | 120.6 | 134.0 | 0.900 [0.848, 0.931] | 3/8 | 0.0% | 27.2 |
| run-29184/lane3 | adaptive_headroom2 | 120.0 | 134.0 | 0.896 [0.835, 0.935] | 3/8 | 0.0% | 27.3 |
| run-29184/lane1 | block8 | 113.8 | 133.9 | 0.850 [0.744, 0.939] | 4/8 | 0.0% | 28.0 |
| run-29184/lane3 | branch_b8_first | 111.4 | 134.0 | 0.831 [0.727, 0.911] | 8/8 | 0.0% | 28.4 |
| run-29184/lane3 | branch_b8_weakest | 108.5 | 134.0 | 0.810 [0.713, 0.877] | 8/8 | 0.0% | 28.8 |
| run-29184/lane2 | adaptive_progress | 90.9 | 134.1 | 0.678 [0.534, 0.827] | 4/8 | 0.0% | 31.8 |
| run-29184/lane0 | verify4 | 87.2 | 133.6 | 0.652 [0.484, 0.837] | 5/8 | 0.0% | 32.6 |
| run-29184/lane1 | block4 | 83.6 | 133.9 | 0.624 [0.486, 0.774] | 5/8 | 0.0% | 33.5 |

Failed hypotheses: 0. Error traces are retained in radical-summary.json and the raw run folders.
