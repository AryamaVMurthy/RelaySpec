# Short native calibration screen

| Records | Updates | Train error | Validation error | Throughput /512-record reference |
|---:|---:|---:|---:|---:|
|16|128|0.00808|0.03770|0.9931 [0.9506, 1.0268]|
|16|1024|0.00050|0.04863|0.9861 [0.9487, 1.0177]|
|128|128|0.02379|0.02898|1.0103 [0.9859, 1.0296]|
|128|1024|0.02143|0.03139|1.0074 [0.9955, 1.0183]|

Native-column initialized compact maps with16/128 calibration records and128/1024 updates. All non-budget fitting settings match wave42. Eight exposed GSM8K questions512token cap, same512-record compact reference. Development screen, not fresh confirmation or a minimum-data guarantee. Timing excludes original drafter training and cached feature extraction.

Matched short random-initialization controls and a second inherited fitting seed are running in wave45. Do not interpret the earlier long-fit contrast as a budget-independent initialization effect.
