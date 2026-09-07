# Two-scalar versus full native calibration

| Task | Records | Scalar LR | Scalar / crop | Scalar / full fit at same records |
|---|---:|---:|---:|---:|
|GSM8K|16|0.003|0.9991 [0.9823, 1.0229]|0.9713 [0.9306, 1.0066]|
|GSM8K|16|0.03|0.9993 [0.9876, 1.0158]|0.9715 [0.9281, 1.0097]|
|GSM8K|128|0.003|0.9986 [0.9820, 1.0228]|0.9514 [0.8956, 0.9974]|
|GSM8K|128|0.03|1.0061 [0.9860, 1.0375]|0.9585 [0.9053, 1.0046]|
|MATH|16|0.003|1.0061 [0.9849, 1.0207]|0.9662 [0.9361, 0.9888]|
|MATH|16|0.03|1.0035 [0.9859, 1.0164]|0.9638 [0.9320, 0.9864]|
|MATH|128|0.003|1.0043 [0.9948, 1.0112]|0.9574 [0.9265, 0.9794]|
|MATH|128|0.03|1.0080 [0.9905, 1.0208]|0.9610 [0.9274, 0.9848]|
|Code|16|0.003|1.0046 [0.9933, 1.0199]|0.9666 [0.9309, 1.0003]|
|Code|16|0.03|0.9976 [0.9892, 1.0067]|0.9599 [0.9248, 0.9928]|
|Code|128|0.003|1.0001 [0.9892, 1.0153]|0.9543 [0.9299, 0.9761]|
|Code|128|0.03|1.0002 [0.9929, 1.0083]|0.9543 [0.9279, 0.9772]|
|Dialogue|16|0.003|0.9976 [0.9902, 1.0032]|1.0047 [0.9872, 1.0158]|
|Dialogue|16|0.03|1.0019 [0.9964, 1.0084]|1.0090 [0.9922, 1.0250]|
|Dialogue|128|0.003|1.0002 [0.9915, 1.0067]|0.9921 [0.9824, 1.0104]|
|Dialogue|128|0.03|1.0000 [0.9941, 1.0056]|0.9919 [0.9802, 1.0124]|

Nine methods per worker: full repacked native, untrained cropped native, original512-record compact reference,16/128-record full inherited fits and all four two-scalar fits. Scalar arms use128 updates at rates0.003/0.03, full fits128 updates at0.0006. Eight exposed requests per workload512token cap, with two turns per dialogue conversation. Paired intervals cluster conversations and do not adjust for development selection or multiple comparisons. No new quality confirmation.
