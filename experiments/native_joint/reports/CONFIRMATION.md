# Frozen128-request confirmation

All128 requests completed:32 each GSM8K, MATH500, HumanEval and MTBench, output cap2048, two timing repeats. Full wall time includes prefill. AR is a single-generation quality reference outside repeated timing. No configuration changed from the frozen selection.

| Method | TPS | / Original [95% interval] | / Full DDTree [95% interval] | Exact original | Exact AR | Capped |
|---|---:|---|---|---:|---:|---:|
| native | 169.5 | 1.000 [1.000, 1.000] | 0.850 [0.835, 0.866] | 128/128 | 40/128 | 1/128 |
| compact_linear | 159.2 | 0.939 [0.930, 0.948] | 0.799 [0.783, 0.814] | 128/128 | 40/128 | 1/128 |
| candidate | 185.9 | 1.097 [1.070, 1.123] | 0.933 [0.916, 0.950] | 42/128 | 40/128 | 1/128 |
| reference | 199.3 | 1.176 [1.155, 1.198] | 1.000 [1.000, 1.000] | 41/128 | 42/128 | 1/128 |

`compact_linear` isolates joint training; `candidate` is that checkpoint with DDTree47; `reference` is the released full drafter with DDTree63. DDTree is existing prior art. Output identity is measured, not assumed; see the separate quality audit before making accuracy claims. Intervals resample paired questions within workload, not independent timing repeats; they exclude training-seed variation.
