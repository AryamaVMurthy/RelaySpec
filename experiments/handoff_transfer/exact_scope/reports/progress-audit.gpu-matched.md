GPU-matched comparisons verified: **6/21**.

Each method is compared with references on the same physical GPU, using the same 128 requests, 2048-token cap, serving batch and repetition index. Three timing repetitions and one fitting seed. These remain development measurements, not untouched confirmation results.

The retained CE/AUF fits use one token-loss refinement epoch. Warm-start calibration is separate: q8 ZIP initialization: 4096 records, 3 calibration epochs; llama ZIP initialization: 4096 records, 3 calibration epochs; cross ZIP initialization: 4096 records, 1 calibration epochs. Original-interface CE uses its separate original-MSE initializer. Refinement timings alone are not total calibration costs. The six completed feature-loss cells are historical artifacts, outside the primary matrix.

| Family | Method | Mean TPS | / AR | / Original | / ZIP |
|---|---|---:|---:|---:|---:|
| cross | cross/five_maps/ce | 1443.6 | 1.305 | 1.288 | 1.364 |
| llama | llama/dense_fusion/ce | 4585.2 | 2.895 | 1.118 | 1.101 |
| llama | llama/five_ba56/auf | 4579.2 | 2.891 | 1.117 | 1.099 |
| llama | llama/five_maps/ce | 4546.2 | 2.870 | 1.109 | 1.091 |
| llama | llama/normal_ce/ce | 4541.3 | 2.867 | 1.107 | 1.090 |
| q8 | q8/five_maps/ce | 2685.7 | 2.185 | 0.996 | 0.998 |
