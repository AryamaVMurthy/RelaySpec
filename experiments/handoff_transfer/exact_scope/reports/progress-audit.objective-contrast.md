Matched data/initializer/architecture/optimizer settings, one fitting seed per pair. Three timing passes are not three fitting replications. Accepted proposal tokens per draft block exclude the verifier bonus; this is not a throughput or universal accuracy guarantee.

| Family | Interface | CE accepted/block | AUF accepted/block | Accepted/block gain | Fewer draft blocks |
|---|---|---:|---:|---:|---:|
| q8 | five_maps | 5.9764 | 6.1000 | 2.07% | 1.72% |
| q8 | dense_fusion | 6.3237 | 6.4714 | 2.34% | 1.97% |
| q8 | five_ba56 | 6.3030 | 6.4329 | 2.06% | 1.75% |
| llama | five_maps | 4.3076 | 4.3975 | 2.09% | 1.69% |
| llama | dense_fusion | 4.3465 | 4.4412 | 2.18% | 1.75% |
| llama | five_ba56 | 4.2529 | 4.3565 | 2.44% | 1.92% |
| cross | five_maps | 4.7938 | 5.0750 | 5.86% | 4.64% |
| cross | dense_fusion | 3.9852 | 4.2425 | 6.46% | 4.90% |
| cross | five_ba56 | 3.9913 | 4.2534 | 6.57% | 4.96% |
