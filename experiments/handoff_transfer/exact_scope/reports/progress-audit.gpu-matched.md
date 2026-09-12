GPU-matched comparisons verified: **19/21**.

Each method is compared with references on the same physical GPU, using the same 128 requests, 2048-token cap, serving batch and repetition index. Three timing repetitions and one fitting seed. These remain development measurements, not untouched confirmation results.

The retained CE/AUF fits use one token-loss refinement epoch. Warm-start calibration is separate: q8 ZIP initialization: 4096 records, 3 calibration epochs; llama ZIP initialization: 4096 records, 3 calibration epochs; cross ZIP initialization: 4096 records, 1 calibration epochs. Original-interface CE uses its separate original-MSE initializer. Refinement timings alone are not total calibration costs. The six completed feature-loss cells are historical artifacts, outside the primary matrix.

| Family | Method | Mean TPS | / AR | / Original | / ZIP |
|---|---|---:|---:|---:|---:|
| cross | cross/dense_fusion/ce | 1209.6 | 1.084 | 1.084 | 1.158 |
| cross | cross/five_ba56/auf | 1307.0 | 1.172 | 1.171 | 1.251 |
| cross | cross/five_ba56/ce | 1248.7 | 1.119 | 1.131 | 1.213 |
| cross | cross/five_maps/auf | 1531.0 | 1.372 | 1.386 | 1.487 |
| cross | cross/five_maps/ce | 1443.6 | 1.305 | 1.288 | 1.364 |
| llama | llama/dense_fusion/auf | 4651.7 | 2.932 | 1.123 | 1.108 |
| llama | llama/dense_fusion/ce | 4585.2 | 2.895 | 1.118 | 1.101 |
| llama | llama/five_ba56/auf | 4579.2 | 2.891 | 1.117 | 1.099 |
| llama | llama/five_ba56/ce | 4479.7 | 2.823 | 1.082 | 1.067 |
| llama | llama/five_maps/auf | 4672.5 | 2.945 | 1.128 | 1.113 |
| llama | llama/five_maps/ce | 4546.2 | 2.870 | 1.109 | 1.091 |
| llama | llama/normal_ce/ce | 4541.3 | 2.867 | 1.107 | 1.090 |
| q8 | q8/dense_fusion/auf | 2844.8 | 2.300 | 1.070 | 1.064 |
| q8 | q8/dense_fusion/ce | 2842.6 | 2.298 | 1.028 | 1.027 |
| q8 | q8/five_ba56/auf | 2822.1 | 2.282 | 1.061 | 1.055 |
| q8 | q8/five_ba56/ce | 2820.0 | 2.280 | 1.020 | 1.019 |
| q8 | q8/five_maps/auf | 2788.6 | 2.254 | 1.009 | 1.007 |
| q8 | q8/five_maps/ce | 2685.7 | 2.185 | 0.996 | 0.998 |
| q8 | q8/normal_ce/ce | 2758.2 | 2.230 | 1.037 | 1.031 |
