# Native verification-error correction heads

These heads modify native draft hidden states before the frozen vocabulary projection. Training uses actual native rollout labels only through the first rejection; the frozen target verifies every proposed token at inference. A differentiable soft-prefix objective is a surrogate, not a measured acceptance probability.

Throughput includes full generation wall time, including prefill and correction overhead. Each fit is screened on two requests per GSM8K, MATH, HumanEval and MTBench, a 512-token cap, and two rotated timing repeats. Paired intervals resample requests, clustering timing repeats; they exclude fitting-seed and adaptive-selection uncertainty. The 128-request confirmation remains reserved.

| Run/lane | Head | Train questions | Candidate / native TPS | Ratio [95% interval] | Validation errors repaired | Previously correct broken | Seconds |
|---|---|---:|---:|---|---:|---:|---:|
| run-29228/lane0 | progress_r16_e1 | 64 | 121.3 / 121.5 | 0.998 [0.988, 1.004] | 36/451 | 70/3536 | 105 |
| run-29228/lane1 | progress_r64_e1 | 64 | 120.2 / 121.8 | 0.987 [0.977, 0.994] | 49/451 | 87/3536 | 105 |
| run-29228/lane2 | progress_r64_e4 | 64 | 118.3 / 121.8 | 0.972 [0.948, 0.983] | 59/451 | 117/3536 | 106 |
| run-29228/lane3 | progress_r64_e4 | 64 | 110.0 / 121.6 | 0.905 [0.867, 0.926] | 88/451 | 301/3536 | 110 |
| run-29235/lane0 | first_position_r16 | 64 | 119.8 / 121.2 | 0.988 [0.986, 0.993] | 5/43 | 18/492 | 107 |
| run-29235/lane1 | first_position_r64 | 64 | 120.5 / 121.8 | 0.989 [0.983, 0.993] | 4/43 | 18/492 | 107 |
| run-29235/lane2 | margin_preservation_1.0 | 64 | 119.1 / 121.3 | 0.982 [0.951, 1.003] | 49/451 | 86/3536 | 110 |
| run-29235/lane3 | margin_preservation_10.0 | 64 | 121.3 / 121.6 | 0.998 [0.984, 1.006] | 29/451 | 42/3536 | 110 |
| run-29271/lane0 | mixed_progress_ce_r16 | 192 | 119.4 / 120.5 | 0.991 [0.978, 0.998] | 67/2185 | 116/8318 | 112 |
| run-29271/lane1 | mixed_progress_margin_r64 | 192 | 119.7 / 120.7 | 0.992 [0.979, 1.011] | 70/2185 | 99/8318 | 120 |
| run-29282/lane0 | soft_progress_t0.1 | 192 | 120.2 / 121.1 | 0.993 [0.973, 1.003] | 75/2185 | 117/8318 | 115 |
| run-29282/lane1 | soft_progress_t0.5 | 192 | 120.4 / 121.3 | 0.993 [0.975, 1.005] | 101/2185 | 138/8318 | 114 |

Protocol note: jobs 29228 and 29235 did not remove post-EOS cached training positions. From job 29271, the EOS label remains supervised but later positions are masked; 159 positions are removed from the mixed cache. Their recorded decoding TPS remains valid, but these fitting protocols are not perfectly matched. Mixed fits also switch to question-uniform sampling and include code and general instructions. Repair/break counts come from different validation pools and must not be compared as absolute rates across pools.

A repaired first rejection can extend a draft, while breaking an earlier accepted token can shorten it. Counts diagnose this tradeoff but do not establish its causal contribution to runtime. The recorded full decoding measurements decide promotion.
