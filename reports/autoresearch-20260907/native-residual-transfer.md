# Low-rank native calibration transfer

|Task|Rank|Learning rate|Residual / crop (95% CI)|Residual / full fit (95% CI)|
|---|---:|---:|---:|---:|
|GSM8K|16|0.0006|1.0278 [1.0017,1.0660]|0.9797 [0.9522,1.0011]|
|GSM8K|16|0.006|1.0127 [0.9832,1.0497]|0.9653 [0.9263,0.9974]|
|GSM8K|128|0.0006|1.0356 [0.9974,1.0913]|0.9871 [0.9720,1.0031]|
|GSM8K|128|0.006|1.0547 [1.0132,1.1043]|1.0053 [0.9759,1.0264]|
|MATH|16|0.0006|0.9970 [0.9722,1.0161]|0.9499 [0.9217,0.9725]|
|MATH|16|0.006|1.0207 [0.9966,1.0462]|0.9725 [0.9473,0.9986]|
|MATH|128|0.0006|1.0154 [0.9835,1.0419]|0.9674 [0.9408,0.9945]|
|MATH|128|0.006|1.0455 [1.0272,1.0727]|0.9961 [0.9739,1.0236]|
|Code|16|0.0006|0.9947 [0.9561,1.0405]|0.9491 [0.9197,0.9746]|
|Code|16|0.006|0.9948 [0.9669,1.0190]|0.9493 [0.9186,0.9725]|
|Code|128|0.0006|1.0080 [0.9745,1.0437]|0.9618 [0.9253,0.9920]|
|Code|128|0.006|1.0449 [1.0100,1.0828]|0.9970 [0.9708,1.0176]|
|Dialogue|16|0.0006|0.9982 [0.9728,1.0135]|0.9899 [0.9662,1.0005]|
|Dialogue|16|0.006|1.0119 [0.9826,1.0291]|1.0034 [0.9793,1.0135]|
|Dialogue|128|0.0006|1.0082 [0.9868,1.0222]|0.9997 [0.9828,1.0177]|
|Dialogue|128|0.006|1.0110 [0.9930,1.0268]|1.0026 [0.9961,1.0156]|

All four rank/rate residual settings with128 calibration records and128 updates, compared in each worker to full-matrix fitting at128 records/128 updates/lr0.0006, untrained crop, original512-record compact and repacked native. Eight exposed requests per workload512token cap, with dialogue turns clustered by conversation. No new confirmation, uncapped quality, equal training time or deployed-size reduction claim. No multiple-comparison adjustment.
