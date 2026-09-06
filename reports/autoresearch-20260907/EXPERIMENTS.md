# Autoresearch experiment registry

All collected terminal lanes, including failures and negative results. Missing lanes are not assumed completed. Repeated baselines and overlapping development questions are not independent experiments. Per-lane confirmation summaries must not replace the aggregated fixed analyses.

| Wave / lane | Job | Family / transform | Status | Seconds | Requests | Candidate throughput retention |
|---|---:|---|---|---:|---:|---|
| 1 / 0 | 28391 | dflash / svd | pass | 120.1 | 8 | relay_svd1024: 93.9%; relay_svd256: 27.5%; relay_svd512: 47.8% |
| 1 / 1 | 28391 | dflash / drop_taps | pass | 115.7 | 8 | relay_drop1: 94.8%; relay_drop17: 100.4%; relay_drop25: 74.7%; relay_drop33: 71.4%; relay_drop9: 100.0% |
| 1 / 2 | 28391 | dflash / data_interpolation | pass | 117.1 | 8 | relay_blend25: 69.1%; relay_blend50: 86.9%; relay_blend75: 97.1%; relay_small: 53.3% |
| 1 / 3 | 28391 | eagle3 / svd | pass | 173.9 | 8 | relay_svd1024: 51.5%; relay_svd256: 27.4%; relay_svd512: 34.6% |
| 2 / 0 | 28392 | dflash / joint_drop | pass | 120.4 | 8 | relay_drop_1_9_17: 87.4%; relay_drop_1_9_17_25: 43.8%; relay_drop_1_9_17_33: 49.1%; relay_drop_9_17: 92.5% |
| 2 / 1 | 28392 | eagle3 / joint_drop | pass | 158.6 | 8 | relay_drop_1_9_17: 81.8%; relay_drop_1_9_17_25: 39.5%; relay_drop_1_9_17_33: 47.1%; relay_drop_9_17: 89.1% |
| 2 / 2 | 28392 | dflash / quantization_noise | pass | 120.9 | 8 | relay_fakeq2: 22.2%; relay_fakeq4: 78.8%; relay_fakeq8: 101.1% |
| 2 / 3 | 28392 | dflash / native_svd | failed | 11.1 | — | — |
| 3 / 0 | 28393 | dflash / fit_reduced_taps | pass | 139.1 | 8 | relay_reduced: 98.7% |
| 3 / 1 | 28393 | dflash / fit_reduced_taps | pass | 134.5 | 8 | relay_reduced: 82.5% |
| 3 / 2 | 28393 | eagle3 / fit_reduced_taps | pass | 169.5 | 8 | relay_reduced: 93.5% |
| 3 / 3 | 28393 | dflash / native_svd | pass | 110.6 | 8 | relay_svd1024: 96.0%; relay_svd256: 34.5%; relay_svd512: 71.5% |
| 4 / 0 | 28400 | dflash / existing | pass | 253.8 | 32 | relay_factor1024: 96.9%; relay_last2: 97.7% |
| 4 / 1 | 28400 | dflash / existing | pass | 227.1 | 32 | relay_factor1024: 100.8%; relay_last2: 101.6% |
| 4 / 2 | 28400 | eagle3 / existing | pass | 317.7 | 32 | relay_last2: 96.3% |
| 4 / 3 | 28400 | dflash / native_svd | pass | 287.3 | 32 | relay_svd1024: 93.8%; relay_svd1536: 99.1%; relay_svd2048: 100.9% |
| 5 / 0 | 28413 | dflash / fit_reduced_taps | pass | 274.4 | 16 | relay_reduced: 99.3% |
| 5 / 1 | 28413 | eagle3 / fit_reduced_taps | pass | 342.0 | 16 | relay_reduced: 93.7% |
| 5 / 2 | 28413 | dflash / fit_reduced_taps | pass | 199.0 | 16 | relay_reduced: 97.6% |
| 5 / 3 | 28413 | eagle3 / fit_reduced_taps | pass | 240.9 | 16 | relay_reduced: 94.8% |
| 6 / 0 | 28414 | dflash / existing | pass | 86.2 | 16 | relay_last2: 96.6% |
| 6 / 1 | 28414 | dflash / existing | pass | 102.1 | 16 | relay_last2: 95.9% |
| 6 / 2 | 28414 | dflash / existing | pass | 80.2 | 16 | relay_last2: 99.8% |
| 6 / 3 | 28414 | dflash / existing | pass | 74.8 | 16 | relay_last2: 99.2% |
| 7 / 0 | 28416 | dflash / existing | pass | 112.7 | 16 | relay_svd1536: 99.4% |
| 7 / 1 | 28416 | dflash / existing | pass | 126.3 | 16 | relay_svd1536: 99.6% |
| 7 / 2 | 28416 | dflash / existing | pass | 96.7 | 16 | relay_svd1536: 98.7% |
| 7 / 3 | 28416 | dflash / existing | pass | 98.7 | 16 | relay_svd1536: 100.3% |
| 8 / 0 | 28419 | dflash / activation_svd | pass | 118.0 | 8 | relay_activation1024: 95.8%; relay_activation512: 72.4%; relay_weight1024: 93.6%; relay_weight512: 47.6% |
| 8 / 1 | 28419 | eagle3 / activation_svd | pass | 162.8 | 8 | relay_activation1024: 90.4%; relay_activation512: 81.2%; relay_weight1024: 52.1%; relay_weight512: 34.5% |
| 8 / 2 | 28419 | dflash / activation_svd | pass | 113.8 | 8 | relay_activation1024: 92.4%; relay_activation512: 73.3%; relay_weight1024: 96.4%; relay_weight512: 71.5% |
| 8 / 3 | 28419 | dflash / activation_svd | pass | 144.6 | 8 | relay_activation1024: 93.6%; relay_activation512: 72.8%; relay_weight1024: 90.7%; relay_weight512: 50.3% |
| 9 / 0 | 28421 | dflash / fit_reduced_taps | pass | 212.5 | 16 | relay_reduced: 56.3% |
| 9 / 1 | 28421 | dflash / fit_reduced_taps | pass | 202.6 | 16 | relay_reduced: 91.5% |
| 9 / 2 | 28421 | eagle3 / fit_reduced_taps | pass | 257.0 | 16 | relay_reduced: 57.0% |
| 9 / 3 | 28421 | eagle3 / fit_reduced_taps | pass | 246.5 | 16 | relay_reduced: 91.4% |
| 10 / 0 | 28425 | eagle3 / fit_reduced_taps | timeout | 540.0 | — | — |
| 10 / 1 | 28425 | eagle3 / existing | pass | 353.8 | 32 | relay_activation1024: 97.2%; relay_weight1024: 59.3% |
| 10 / 2 | 28425 | dflash / existing | pass | 353.4 | 32 | relay_activation1024: 96.6%; relay_trained1024: 97.3%; relay_weight1024: 92.7% |
| 10 / 3 | 28425 | dflash / quantization_noise | pass | 388.6 | 32 | relay_fakeq2: 23.0%; relay_fakeq4: 90.3%; relay_fakeq8: 100.3% |
| 11 / 0 | 28427 | dflash / existing | pass | 183.4 | 8 | relay_early: 40.5%; relay_last2: 99.7%; relay_spaced: 84.8% |
| 11 / 1 | 28427 | dflash / existing | pass | 267.1 | 16 | relay_early: 65.8%; relay_last2: 97.8%; relay_spaced: 91.3% |
| 11 / 2 | 28427 | eagle3 / existing | pass | 251.4 | 8 | relay_early: 52.0%; relay_last2: 85.4%; relay_spaced: 84.9% |
| 11 / 3 | 28427 | eagle3 / existing | pass | 379.3 | 16 | relay_early: 69.6%; relay_last2: 91.9%; relay_spaced: 93.0% |
| 12 / 0 | 28428 | eagle3 / existing | pass | 416.4 | 32 | relay_activation1024: 93.9%; relay_reduced: 96.1%; relay_weight1024: 54.1% |
| 12 / 1 | 28428 | eagle3 / existing | pass | 399.7 | 32 | relay_activation1024: 96.7%; relay_trained1024: 96.5%; relay_weight1024: 59.4% |
| 12 / 2 | 28428 | dflash / fit_reduced_taps | pass | 182.7 | 8 | relay_reduced: 97.3% |
| 12 / 3 | 28428 | dflash / fit_reduced_taps | pass | 143.9 | 8 | relay_reduced: 84.2% |
| 13 / 0 | 28429 | dflash / existing | pass | 180.6 | 32 | relay_one: 84.3%; relay_svd1536: 98.3%; relay_two: 97.2% |
| 13 / 1 | 28429 | dflash / existing | pass | 146.3 | 32 | relay_one: 88.9%; relay_svd1536: 98.3%; relay_two: 97.3% |
| 13 / 2 | 28429 | dflash / existing | pass | 95.3 | 8 | relay_one: 72.1%; relay_svd1536: 99.4%; relay_two: 97.9% |
| 13 / 3 | 28429 | dflash / existing | pass | 180.9 | 16 | relay_one: 87.3%; relay_svd1536: 99.5%; relay_two: 98.2% |
| 14 / 0 | 28430 | dflash / fit_reduced_taps | pass | 338.4 | 32 | relay_reduced: 98.3% |
| 14 / 1 | 28430 | dflash / fit_reduced_taps | pass | 324.2 | 32 | relay_reduced: 100.4% |
| 14 / 2 | 28430 | dflash / fit_reduced_taps | pass | 331.2 | 32 | relay_reduced: 97.0% |
| 14 / 3 | 28430 | dflash / fit_reduced_taps | pass | 339.4 | 32 | relay_reduced: 97.7% |
| 15 / 0 | 28431 | dflash / existing | pass | 123.9 | 32 | relay_math: 97.3%; relay_mixed: 98.3% |
| 15 / 1 | 28431 | dflash / existing | pass | 152.8 | 16 | relay_math: 97.9%; relay_mixed: 108.4% |
| 15 / 2 | 28431 | dflash / existing | pass | 123.6 | 32 | relay_math: 97.1%; relay_mixed: 98.5% |
| 15 / 3 | 28431 | dflash / existing | pass | 149.3 | 16 | relay_math: 98.2%; relay_mixed: 99.3% |
| 16 / 0 | 28432 | dflash / existing | pass | 167.0 | 32 | relay_math: 98.3%; relay_mixed: 100.4% |
| 16 / 1 | 28432 | dflash / existing | pass | 153.9 | 32 | relay_math: 97.6%; relay_mixed: 96.9% |
| 16 / 2 | 28432 | dflash / existing | pass | 146.3 | 32 | relay_math: 97.0%; relay_mixed: 97.6% |
| 16 / 3 | 28432 | dflash / existing | pass | 147.7 | 32 | relay_math: 97.0%; relay_mixed: 98.2% |
