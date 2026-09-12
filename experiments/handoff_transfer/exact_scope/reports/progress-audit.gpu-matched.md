GPU-matched comparisons verified: **5/30**.

Each method is compared with references on the same physical GPU, using the same 128 requests, 2048-token cap, serving batch and repetition index. Three timing repetitions and one fitting seed. These remain development measurements, not untouched confirmation results.

| Family | Method | Mean TPS | / AR | / Original | / ZIP |
|---|---|---:|---:|---:|---:|
| llama | llama/dense_fusion/ce | 4585.2 | 2.895 | 1.118 | 1.101 |
| llama | llama/five_ba56/auf | 4579.2 | 2.891 | 1.117 | 1.099 |
| llama | llama/five_maps/ce | 4546.2 | 2.870 | 1.109 | 1.091 |
| llama | llama/normal_ce/ce | 4541.3 | 2.867 | 1.107 | 1.090 |
| q8 | q8/five_maps/ce | 2685.7 | 2.185 | 0.996 | 0.998 |
