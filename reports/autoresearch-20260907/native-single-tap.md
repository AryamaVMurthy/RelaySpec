# Single-layer native initialization development screen

| Records | Initialization | Train error | Validation error | Throughput /512-record compact reference |
|---:|---|---:|---:|---:|
|128|random|0.260163|0.325373|0.5337 [0.4930, 0.5697]|
|128|native_columns|0.163076|0.184779|0.6426 [0.6062, 0.6732]|
|128|random|0.099164|0.158321|0.8004 [0.7506, 0.8382]|
|128|native_columns|0.085016|0.131942|0.8561 [0.7968, 0.9068]|

Matched128-record single-layer random versus inherited native initialization,128/1024 updates. Eight exposed GSM8K requests512token cap, with original512-record two-layer map as reference. No fresh confirmation.
