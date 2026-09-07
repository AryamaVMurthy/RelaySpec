# Native initialization development screen

| Records | Initialization | Train error | Validation error | Throughput /512-record compact reference |
|---:|---|---:|---:|---:|
|16|random|0.097540|0.368630|0.5064 [0.4893, 0.5253]|
|16|native_columns|0.009549|0.045871|0.9319 [0.8975, 0.9716]|
|128|random|0.225838|0.281350|0.6245 [0.6036, 0.6476]|
|128|native_columns|0.036856|0.042742|0.9340 [0.9097, 0.9674]|

Native EAGLE3 late-two-layer map with16/128 records and128 updates. Matched random versus inherited native columns, raw post-fusion interface without external norm. Eight exposed GSM8K requests512token cap, paired full repacked native reference. No cross-family absolute feature-error comparison, no fresh confirmation.
