# Low-rank native calibration transfer

|Task|Rank|Seed|Residual / crop (95% CI)|Residual / full fit (95% CI)|
|---|---:|---:|---:|---:|
|GSM8K|16|1730|1.0163 [0.9884,1.0545]|0.9685 [0.9193,0.9979]|
|GSM8K|128|1730|1.0363 [0.9950,1.0889]|0.9876 [0.9536,1.0138]|
|GSM8K|16|1731|1.0259 [0.9931,1.0673]|0.9777 [0.9300,1.0111]|
|GSM8K|128|1731|1.0339 [0.9875,1.0810]|0.9853 [0.9473,1.0222]|
|MATH|16|1730|1.0173 [0.9838,1.0510]|0.9686 [0.9396,0.9982]|
|MATH|128|1730|1.0485 [1.0177,1.0771]|0.9983 [0.9794,1.0163]|
|MATH|16|1731|1.0088 [0.9827,1.0364]|0.9605 [0.9397,0.9794]|
|MATH|128|1731|1.0414 [1.0160,1.0762]|0.9915 [0.9842,0.9985]|
|Code|16|1730|1.0065 [0.9742,1.0334]|0.9600 [0.9241,0.9883]|
|Code|128|1730|1.0340 [0.9929,1.0737]|0.9862 [0.9529,1.0056]|
|Code|16|1731|1.0119 [0.9775,1.0457]|0.9652 [0.9336,0.9891]|
|Code|128|1731|1.0203 [0.9938,1.0585]|0.9731 [0.9546,0.9904]|
|Dialogue|16|1730|1.0060 [0.9741,1.0231]|0.9955 [0.9695,1.0065]|
|Dialogue|128|1730|1.0124 [0.9906,1.0232]|1.0018 [0.9928,1.0061]|
|Dialogue|16|1731|1.0035 [0.9762,1.0183]|0.9930 [0.9710,1.0015]|
|Dialogue|128|1731|1.0104 [0.9841,1.0230]|0.9998 [0.9817,1.0097]|

All four rank16/128 and seed1730/1731 settings at lr0.006 with128 calibration records and128 updates, compared in each worker to full-matrix fitting at128 records/128 updates/lr0.0006, untrained crop, original512-record compact and repacked native. Eight exposed requests per workload512token cap, with dialogue turns clustered by conversation. No new confirmation, uncapped quality, equal training time or deployed-size reduction claim. No multiple-comparison adjustment.
