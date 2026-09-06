# Native initialization development screen

| Records | Initialization | Train error | Validation error | Throughput /512-record compact reference |
|---:|---|---:|---:|---:|
|16|random|0.000875|0.233232|0.6699 [0.6289, 0.7105]|
|16|native_columns|0.000145|0.055909|1.0071 [0.9725, 1.0366]|
|128|random|0.024815|0.042781|0.9669 [0.9401, 1.0015]|
|128|native_columns|0.018059|0.033988|1.0059 [0.9832, 1.0346]|

Native compact late-layer mapper, matched16/128-record random versus released-column initialization. Seed1729,8192 updates, unchanged teacher/width/normalization and record prefixes. Eight exposed GSM8K questions512token cap, each compared with original512-record compact reference. Untrained cropping and cross-domain controls pending wave43. No fresh confirmation, minimum-data threshold, or standalone novelty claim.
