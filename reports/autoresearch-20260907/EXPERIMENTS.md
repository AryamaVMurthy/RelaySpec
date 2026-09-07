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
| 17 / 0 | 28433 | dflash / fit_reduced_taps | pass | 272.7 | 16 | relay_math_seed1729: 98.5%; relay_mixed_seed1729: 108.9%; relay_reduced: 98.1% |
| 17 / 1 | 28433 | dflash / fit_reduced_taps | pass | 271.9 | 16 | relay_math_seed1729: 99.0%; relay_mixed_seed1729: 109.7%; relay_reduced: 110.0% |
| 17 / 2 | 28433 | dflash / fit_reduced_taps | pass | 284.4 | 16 | relay_math_seed1729: 98.2%; relay_mixed_seed1729: 99.3%; relay_reduced: 98.6% |
| 17 / 3 | 28433 | dflash / fit_reduced_taps | pass | 291.2 | 16 | relay_math_seed1729: 98.2%; relay_mixed_seed1729: 99.3%; relay_reduced: 98.7% |
| 18 / 0 | 28434 | dflash / existing | pass | 304.1 | 16 | relay_math_seed1729: 98.1%; relay_math_seed1730: 98.0%; relay_mixed_seed1729: 106.8%; relay_mixed_seed1730: 106.3% |
| 18 / 1 | 28434 | dflash / existing | pass | 276.3 | 16 | relay_math_seed1729: 97.6%; relay_math_seed1730: 97.7%; relay_mixed_seed1729: 103.6%; relay_mixed_seed1730: 104.0% |
| 18 / 2 | 28434 | dflash / existing | pass | 271.7 | 16 | relay_math_seed1729: 97.1%; relay_math_seed1730: 98.3%; relay_mixed_seed1729: 99.7%; relay_mixed_seed1730: 99.2% |
| 18 / 3 | 28434 | dflash / existing | pass | 259.9 | 16 | relay_math_seed1729: 97.5%; relay_math_seed1730: 97.4%; relay_mixed_seed1729: 97.0%; relay_mixed_seed1730: 97.0% |
| 19 / 0 | 28435 | dflash / fit_reduced_taps | pass | 139.1 | 8 | relay_early: 56.6%; relay_last2: 98.6%; relay_reduced: 56.0% |
| 19 / 1 | 28435 | dflash / fit_reduced_taps | pass | 145.1 | 8 | relay_early: 55.5%; relay_last2: 97.4%; relay_reduced: 50.0% |
| 19 / 2 | 28435 | eagle3 / fit_reduced_taps | pass | 180.3 | 8 | relay_early: 56.5%; relay_last2: 92.7%; relay_reduced: 57.9% |
| 19 / 3 | 28435 | eagle3 / fit_reduced_taps | pass | 180.0 | 8 | relay_early: 56.5%; relay_last2: 92.6%; relay_reduced: 57.4% |
| 25 / 0 | 28459 | dflash / existing | pass | 70.8 | 8 | — |
| 25 / 1 | 28459 | dflash / existing | pass | 78.3 | 8 | — |
| 25 / 2 | 28459 | dflash / existing | pass | 60.8 | 8 | — |
| 25 / 3 | 28459 | dflash / existing | pass | 65.8 | 8 | — |
| 26 / 0 | 28460 | dflash / existing | pass | 58.8 | 8 | — |
| 26 / 1 | 28460 | dflash / existing | pass | 59.7 | 8 | — |
| 26 / 2 | 28460 | dflash / existing | pass | 80.6 | 16 | — |
| 26 / 3 | 28460 | dflash / existing | pass | 82.1 | 16 | — |
| 27 / 0 | 28464 | dflash / existing | pass | 103.5 | 8 | — |
| 27 / 1 | 28464 | dflash / existing | pass | 178.9 | 8 | — |
| 27 / 2 | 28464 | dflash / existing | pass | 86.7 | 8 | — |
| 27 / 3 | 28464 | dflash / existing | pass | 151.6 | 8 | — |
| 28 / 0 | 28488 | dflash / existing | pass | 107.2 | 16 | — |
| 28 / 1 | 28488 | dflash / existing | pass | 112.2 | 16 | — |
| 28 / 2 | 28488 | dflash / existing | pass | 164.2 | 16 | — |
| 28 / 3 | 28488 | dflash / existing | pass | 174.4 | 16 | — |
| 29 / 0 | 28502 | dflash / existing | pass | 385.9 | 32 | — |
| 29 / 1 | 28502 | dflash / existing | pass | 391.1 | 32 | — |
| 29 / 2 | 28502 | dflash / existing | pass | 297.1 | 16 | — |
| 29 / 3 | 28502 | dflash / existing | pass | 312.8 | 16 | — |
| 30 / 0 | 28520 | dflash / existing | pass | 27.6 | 2 | relay_selected: 100.2% |
| 30 / 1 | 28520 | dflash / existing | pass | 30.0 | 2 | relay_selected: 100.3% |
| 30 / 2 | 28520 | dflash / existing | pass | 35.5 | 2 | relay_selected: 99.6% |
| 30 / 3 | 28520 | dflash / existing | pass | 37.9 | 2 | relay_selected: 100.5% |
| 31 / 0 | 28521 | dflash / existing | pass | 49.5 | 2 | relay_selected: 100.0% |
| 31 / 1 | 28521 | dflash / existing | pass | 53.2 | 2 | relay_selected: 99.9% |
| 31 / 2 | 28521 | dflash / existing | pass | 77.0 | 2 | relay_selected: 100.2% |
| 31 / 3 | 28521 | dflash / existing | pass | 77.5 | 2 | relay_selected: 100.0% |
| 32 / 0 | 28522 | dflash / existing | pass | 51.0 | 2 | relay_selected: 99.9% |
| 32 / 1 | 28522 | dflash / existing | pass | 48.4 | 2 | relay_selected: 99.9% |
| 32 / 2 | 28522 | dflash / existing | pass | 74.1 | 2 | relay_selected: 99.9% |
| 32 / 3 | 28522 | dflash / existing | pass | 74.8 | 2 | relay_selected: 99.8% |
| 33 / 0 | 28523 | dflash / existing | pass | 90.6 | 16 | relay_selected: 100.1% |
| 33 / 1 | 28523 | dflash / existing | pass | 85.5 | 16 | relay_selected: 100.2% |
| 33 / 2 | 28523 | dflash / existing | pass | 158.8 | 32 | relay_selected: 100.4% |
| 33 / 3 | 28523 | dflash / existing | pass | 160.1 | 32 | relay_selected: 100.1% |
| 34 / 0 | 28524 | dflash / existing | pass | 177.8 | 16 | relay_one: 85.3%; relay_svd1536: 98.8%; relay_two: 96.9% |
| 34 / 1 | 28524 | dflash / existing | pass | 172.5 | 16 | relay_one: 83.9%; relay_svd1536: 98.2%; relay_two: 97.5% |
| 34 / 2 | 28524 | dflash / existing | pass | 168.2 | 16 | relay_one: 86.8%; relay_svd1536: 98.5%; relay_two: 97.6% |
| 34 / 3 | 28524 | dflash / existing | pass | 177.9 | 16 | relay_one: 84.9%; relay_svd1536: 99.8%; relay_two: 99.2% |
| 35 / 0 | 28525 | dflash / fit_reduced_taps | pass | 92.0 | 8 | relay_reduced: 52.5% |
| 35 / 1 | 28525 | dflash / fit_reduced_taps | pass | 79.5 | 8 | relay_reduced: 79.9% |
| 35 / 2 | 28525 | dflash / fit_reduced_taps | pass | 80.2 | 8 | relay_reduced: 82.2% |
| 35 / 3 | 28525 | dflash / fit_reduced_taps | pass | 86.7 | 8 | relay_reduced: 60.7% |
| 36 / 0 | 28526 | dflash / fit_reduced_taps | pass | 93.4 | 8 | relay_late_dense: 83.1%; relay_reduced: 81.1% |
| 36 / 1 | 28526 | dflash / fit_reduced_taps | pass | 96.0 | 8 | relay_late_dense: 84.3%; relay_reduced: 84.2% |
| 36 / 2 | 28526 | dflash / fit_reduced_taps | pass | 93.6 | 8 | relay_late_dense: 82.9%; relay_reduced: 83.1% |
| 36 / 3 | 28526 | dflash / fit_reduced_taps | pass | 96.2 | 8 | relay_late_dense: 82.8%; relay_reduced: 83.1% |
| 37 / 0 | 28527 | eagle3 / fit_reduced_taps | pass | 376.9 | 8 | relay_reduced: 104.4% |
| 37 / 1 | 28527 | eagle3 / fit_reduced_taps | pass | 373.4 | 8 | relay_reduced: 100.3% |
| 37 / 2 | 28527 | eagle3 / fit_reduced_taps | pass | 510.4 | 8 | relay_reduced: 106.2% |
| 37 / 3 | 28527 | eagle3 / fit_reduced_taps | pass | 510.4 | 8 | relay_reduced: 105.4% |
| 38 / 0 | 28528 | eagle3 / existing | pass | 136.4 | 8 | relay_lower_rate: 106.9% |
| 38 / 1 | 28528 | eagle3 / existing | pass | 133.9 | 8 | relay_lower_rate: 103.7% |
| 38 / 2 | 28528 | eagle3 / existing | pass | 243.4 | 8 | relay_dense2048: 105.3%; relay_lower_rate: 109.1%; relay_original_matched: 107.3% |
| 38 / 3 | 28528 | eagle3 / existing | pass | 246.4 | 8 | relay_dense2048: 104.9%; relay_lower_rate: 108.1%; relay_original_matched: 103.5% |
| 39 / 0 | 28529 | dflash / existing | pass | 99.9 | 8 | relay_one: 69.3%; relay_svd1536: 101.4%; relay_two: 97.4% |
| 39 / 1 | 28529 | dflash / existing | pass | 117.8 | 8 | relay_one: 76.0%; relay_svd1536: 96.4%; relay_two: 97.2% |
| 39 / 2 | 28529 | dflash / existing | pass | 100.2 | 8 | relay_one: 70.6%; relay_svd1536: 99.1%; relay_two: 94.7% |
| 39 / 3 | 28529 | dflash / existing | pass | 116.4 | 8 | relay_one: 72.0%; relay_svd1536: 98.0%; relay_two: 95.4% |
| 40 / 0 | 28530 | dflash / fit_reduced_taps | pass | 164.8 | 8 | relay_reduced: 66.7% |
| 40 / 1 | 28530 | dflash / fit_reduced_taps | pass | 159.5 | 8 | relay_reduced: 97.2% |
| 40 / 2 | 28530 | dflash / fit_reduced_taps | pass | 151.7 | 8 | relay_reduced: 46.3% |
| 40 / 3 | 28530 | dflash / fit_reduced_taps | pass | 137.2 | 8 | relay_reduced: 87.7% |
| 41 / 0 | 28531 | dflash / fit_reduced_taps | pass | 162.2 | 8 | relay_reduced: 67.0% |
| 41 / 1 | 28531 | dflash / fit_reduced_taps | pass | 159.4 | 8 | relay_reduced: 97.9% |
| 41 / 2 | 28531 | dflash / fit_reduced_taps | pass | 153.9 | 8 | relay_reduced: 47.1% |
| 41 / 3 | 28531 | dflash / fit_reduced_taps | pass | 127.1 | 8 | relay_reduced: 88.9% |
| 42 / 0 | 28532 | dflash / fit_reduced_taps | pass | 166.7 | 8 | relay_reduced: 67.0% |
| 42 / 1 | 28532 | dflash / fit_reduced_taps | pass | 155.0 | 8 | relay_reduced: 100.7% |
| 42 / 2 | 28532 | dflash / fit_reduced_taps | pass | 163.9 | 8 | relay_reduced: 96.7% |
| 42 / 3 | 28532 | dflash / fit_reduced_taps | pass | 164.2 | 8 | relay_reduced: 100.6% |
| 43 / 0 | 28533 | dflash / native_columns | pass | 141.5 | 8 | relay_cropped: 95.7%; relay_n128_native_columns: 100.6%; relay_n128_random: 96.7%; relay_n16_native_columns: 99.8%; relay_n16_random: 66.6%; relay_two: 99.8% |
| 43 / 1 | 28533 | dflash / native_columns | pass | 162.4 | 8 | relay_cropped: 90.8%; relay_n128_native_columns: 96.2%; relay_n128_random: 95.5%; relay_n16_native_columns: 93.5%; relay_n16_random: 69.0%; relay_two: 97.4% |
| 43 / 2 | 28533 | dflash / native_columns | pass | 133.2 | 8 | relay_cropped: 90.6%; relay_n128_native_columns: 94.2%; relay_n128_random: 92.6%; relay_n16_native_columns: 92.6%; relay_n16_random: 52.6%; relay_two: 98.0% |
| 43 / 3 | 28533 | dflash / native_columns | pass | 338.4 | 16 | relay_cropped: 98.4%; relay_n128_native_columns: 98.1%; relay_n128_random: 96.2%; relay_n16_native_columns: 98.4%; relay_n16_random: 72.7%; relay_two: 98.8% |
| 44 / 0 | 28535 | dflash / fit_reduced_taps | pass | 67.4 | 8 | relay_reduced: 99.3% |
| 44 / 1 | 28535 | dflash / fit_reduced_taps | pass | 80.0 | 8 | relay_reduced: 98.6% |
| 44 / 2 | 28535 | dflash / fit_reduced_taps | pass | 71.9 | 8 | relay_reduced: 101.0% |
| 44 / 3 | 28535 | dflash / fit_reduced_taps | pass | 77.3 | 8 | relay_reduced: 100.7% |
| 45 / 0 | 28536 | dflash / fit_reduced_taps | pass | 78.0 | 8 | relay_reduced: 51.7% |
| 45 / 1 | 28536 | dflash / fit_reduced_taps | pass | 70.2 | 8 | relay_reduced: 72.4% |
| 45 / 2 | 28536 | dflash / fit_reduced_taps | pass | 64.8 | 8 | relay_reduced: 98.7% |
| 45 / 3 | 28536 | dflash / fit_reduced_taps | pass | 65.4 | 8 | relay_reduced: 100.7% |
| 46 / 0 | 28537 | dflash / native_columns | pass | 152.7 | 8 | relay_cropped: 95.9%; relay_n128_native_columns: 100.6%; relay_n128_random: 72.4%; relay_n16_native_columns: 98.6%; relay_n16_random: 51.6%; relay_two: 99.9% |
| 46 / 1 | 28537 | dflash / native_columns | pass | 177.8 | 8 | relay_cropped: 90.8%; relay_n128_native_columns: 95.2%; relay_n128_random: 76.2%; relay_n16_native_columns: 94.5%; relay_n16_random: 52.0%; relay_two: 97.4% |
| 46 / 2 | 28537 | dflash / native_columns | pass | 148.6 | 8 | relay_cropped: 90.8%; relay_n128_native_columns: 95.2%; relay_n128_random: 56.2%; relay_n16_native_columns: 94.4%; relay_n16_random: 36.8%; relay_two: 98.2% |
| 46 / 3 | 28537 | dflash / native_columns | pass | 353.6 | 16 | relay_cropped: 98.4%; relay_n128_native_columns: 99.2%; relay_n128_random: 75.0%; relay_n16_native_columns: 97.7%; relay_n16_random: 63.1%; relay_two: 98.8% |
| 47 / 0 | 28538 | dflash / fit_reduced_taps | pass | 69.2 | 8 | relay_reduced: 100.0% |
| 47 / 1 | 28538 | dflash / fit_reduced_taps | pass | 68.5 | 8 | relay_reduced: 99.1% |
| 47 / 2 | 28538 | dflash / fit_reduced_taps | pass | 64.6 | 8 | relay_reduced: 100.0% |
| 47 / 3 | 28538 | dflash / fit_reduced_taps | pass | 68.4 | 8 | relay_reduced: 99.1% |
| 48 / 0 | 28539 | dflash / fit_reduced_taps | pass | 68.9 | 8 | relay_reduced: 95.8% |
| 48 / 1 | 28539 | dflash / fit_reduced_taps | pass | 66.4 | 8 | relay_reduced: 97.4% |
| 48 / 2 | 28539 | dflash / fit_reduced_taps | pass | 67.6 | 8 | relay_reduced: 95.9% |
| 48 / 3 | 28539 | dflash / fit_reduced_taps | pass | 69.8 | 8 | relay_reduced: 96.5% |
| 49 / 0 | 28540 | dflash / native_columns | pass | 166.4 | 8 | relay_cropped: 95.9%; relay_gains_lane0: 95.8%; relay_gains_lane1: 95.8%; relay_gains_lane2: 95.7%; relay_gains_lane3: 96.4%; relay_n128_native_columns: 100.6%; relay_n16_native_columns: 98.6%; relay_two: 99.8% |
| 49 / 1 | 28540 | dflash / native_columns | pass | 199.7 | 8 | relay_cropped: 90.8%; relay_gains_lane0: 91.3%; relay_gains_lane1: 91.1%; relay_gains_lane2: 91.2%; relay_gains_lane3: 91.5%; relay_n128_native_columns: 95.2%; relay_n16_native_columns: 94.5%; relay_two: 97.3% |
| 49 / 2 | 28540 | dflash / native_columns | pass | 148.7 | 8 | relay_cropped: 90.7%; relay_gains_lane0: 91.1%; relay_gains_lane1: 90.4%; relay_gains_lane2: 90.7%; relay_gains_lane3: 90.7%; relay_n128_native_columns: 95.0%; relay_n16_native_columns: 94.2%; relay_two: 98.0% |
| 49 / 3 | 28540 | dflash / native_columns | pass | 401.7 | 16 | relay_cropped: 98.4%; relay_gains_lane0: 98.2%; relay_gains_lane1: 98.6%; relay_gains_lane2: 98.4%; relay_gains_lane3: 98.4%; relay_n128_native_columns: 99.2%; relay_n16_native_columns: 97.7%; relay_two: 98.8% |
| 50 / 0 | 28543 | dflash / existing | pass | 186.8 | 16 | relay_cropped: 94.6%; relay_short128: 97.9%; relay_short16: 96.7% |
| 50 / 1 | 28543 | dflash / existing | pass | 168.8 | 16 | relay_cropped: 92.9%; relay_short128: 96.8%; relay_short16: 94.7% |
| 50 / 2 | 28543 | dflash / existing | pass | 158.0 | 16 | relay_cropped: 92.8%; relay_short128: 97.3%; relay_short16: 95.4% |
| 50 / 3 | 28543 | dflash / existing | pass | 190.7 | 16 | relay_cropped: 93.7%; relay_short128: 97.5%; relay_short16: 96.5% |
| 51 / 0 | 28544 | dflash / fit_reduced_taps | pass | 81.9 | 8 | relay_reduced: 53.4% |
| 51 / 1 | 28544 | dflash / fit_reduced_taps | pass | 78.6 | 8 | relay_reduced: 64.3% |
| 51 / 2 | 28544 | dflash / fit_reduced_taps | pass | 74.1 | 8 | relay_reduced: 80.0% |
| 51 / 3 | 28544 | dflash / fit_reduced_taps | pass | 75.7 | 8 | relay_reduced: 85.6% |
| 52 / 0 | 28545 | dflash / fit_reduced_taps | pass | 64.4 | 8 | relay_reduced: 98.6% |
| 52 / 1 | 28545 | dflash / fit_reduced_taps | pass | 65.1 | 8 | relay_reduced: 97.2% |
| 52 / 2 | 28545 | dflash / fit_reduced_taps | pass | 70.0 | 8 | relay_reduced: 99.4% |
| 52 / 3 | 28545 | dflash / fit_reduced_taps | pass | 63.3 | 8 | relay_reduced: 101.2% |
| 53 / 0 | 28547 | dflash / native_columns | pass | 149.2 | 8 | relay_cropped: 95.9%; relay_n128_native_columns: 100.6%; relay_residual_lane0: 98.6%; relay_residual_lane1: 97.1%; relay_residual_lane2: 99.3%; relay_residual_lane3: 101.2%; relay_two: 99.9% |
| 53 / 1 | 28547 | dflash / native_columns | pass | 174.3 | 8 | relay_cropped: 90.8%; relay_n128_native_columns: 95.3%; relay_residual_lane0: 90.5%; relay_residual_lane1: 92.6%; relay_residual_lane2: 92.2%; relay_residual_lane3: 94.9%; relay_two: 97.4% |
| 53 / 2 | 28547 | dflash / native_columns | pass | 137.6 | 8 | relay_cropped: 90.6%; relay_n128_native_columns: 95.0%; relay_residual_lane0: 90.2%; relay_residual_lane1: 90.2%; relay_residual_lane2: 91.4%; relay_residual_lane3: 94.7%; relay_two: 98.0% |
| 53 / 3 | 28547 | dflash / native_columns | pass | 357.3 | 16 | relay_cropped: 98.4%; relay_n128_native_columns: 99.3%; relay_residual_lane0: 98.3%; relay_residual_lane1: 99.6%; relay_residual_lane2: 99.2%; relay_residual_lane3: 99.5%; relay_two: 98.8% |
| 54 / 0 | 28549 | eagle3 / native_svd | failed | 23.6 | — | — |
| 54 / 1 | 28549 | eagle3 / native_svd | failed | 23.8 | — | — |
| 54 / 2 | 28549 | eagle3 / native_svd | failed | 18.8 | — | — |
| 54 / 3 | 28549 | eagle3 / native_svd | failed | 18.9 | — | — |
| 55 / 0 | 28550 | dflash / fit_reduced_taps | pass | 69.8 | 8 | relay_reduced: 97.7% |
| 55 / 1 | 28550 | dflash / fit_reduced_taps | pass | 71.3 | 8 | relay_reduced: 99.1% |
| 55 / 2 | 28550 | dflash / fit_reduced_taps | pass | 67.3 | 8 | relay_reduced: 100.3% |
| 55 / 3 | 28550 | dflash / fit_reduced_taps | pass | 69.0 | 8 | relay_reduced: 99.3% |
| 56 / 0 | 28551 | eagle3 / native_svd | pass | 40.2 | 4 | relay_svd1536: 81.6% |
| 56 / 1 | 28551 | eagle3 / native_svd | pass | 41.1 | 4 | relay_svd1536: 92.7% |
| 56 / 2 | 28551 | eagle3 / native_svd | pass | 44.7 | 4 | relay_svd1536: 97.4% |
| 56 / 3 | 28551 | eagle3 / native_svd | pass | 63.2 | 8 | relay_svd1536: 81.1% |
| 57 / 0 | 28552 | eagle3 / fit_reduced_taps | pass | 110.2 | 8 | relay_reduced: 50.6% |
| 57 / 1 | 28552 | eagle3 / fit_reduced_taps | pass | 87.2 | 8 | relay_reduced: 93.2% |
| 57 / 2 | 28552 | eagle3 / fit_reduced_taps | pass | 101.5 | 8 | relay_reduced: 62.4% |
| 57 / 3 | 28552 | eagle3 / fit_reduced_taps | pass | 88.1 | 8 | relay_reduced: 93.4% |
| 58 / 0 | 28553 | dflash / native_columns | pass | 147.2 | 8 | relay_cropped: 95.9%; relay_n128_native_columns: 100.6%; relay_residual_lane0: 97.4%; relay_residual_lane1: 99.4%; relay_residual_lane2: 98.4%; relay_residual_lane3: 99.1%; relay_two: 99.8% |
| 58 / 1 | 28553 | dflash / native_columns | pass | 176.7 | 8 | relay_cropped: 91.5%; relay_n128_native_columns: 96.1%; relay_residual_lane0: 93.1%; relay_residual_lane1: 96.0%; relay_residual_lane2: 92.3%; relay_residual_lane3: 95.3%; relay_two: 98.3% |
| 58 / 2 | 28553 | dflash / native_columns | pass | 137.7 | 8 | relay_cropped: 90.6%; relay_n128_native_columns: 95.0%; relay_residual_lane0: 91.2%; relay_residual_lane1: 93.7%; relay_residual_lane2: 91.7%; relay_residual_lane3: 92.5%; relay_two: 98.1% |
| 58 / 3 | 28553 | dflash / native_columns | pass | 359.3 | 16 | relay_cropped: 98.2%; relay_n128_native_columns: 99.2%; relay_residual_lane0: 98.8%; relay_residual_lane1: 99.4%; relay_residual_lane2: 98.5%; relay_residual_lane3: 99.2%; relay_two: 98.8% |
| 59 / 0 | 28554 | eagle3 / native_columns | pass | 196.7 | 8 | relay_cropped: 81.4%; relay_n128_native_columns: 94.7%; relay_n128_random: 62.0%; relay_n16_native_columns: 94.3%; relay_n16_random: 50.9%; relay_svd1536: 88.2% |
| 59 / 1 | 28554 | eagle3 / native_columns | pass | 210.6 | 8 | relay_cropped: 78.8%; relay_n128_native_columns: 92.3%; relay_n128_random: 68.5%; relay_n16_native_columns: 95.2%; relay_n16_random: 55.4%; relay_svd1536: 92.7% |
| 59 / 2 | 28554 | eagle3 / native_columns | pass | 267.6 | 8 | relay_cropped: 82.8%; relay_n128_native_columns: 92.9%; relay_n128_random: 44.8%; relay_n16_native_columns: 94.6%; relay_n16_random: 38.7%; relay_svd1536: 93.2% |
| 59 / 3 | 28554 | eagle3 / native_columns | pass | 450.9 | 16 | relay_cropped: 82.8%; relay_n128_native_columns: 92.6%; relay_n128_random: 56.3%; relay_n16_native_columns: 94.3%; relay_n16_random: 53.2%; relay_svd1536: 87.8% |
| 60 / 0 | 28555 | dflash / existing | pass | 46.1 | 4 | — |
| 60 / 1 | 28555 | dflash / existing | pass | 43.2 | 4 | — |
| 60 / 2 | 28555 | eagle3 / existing | pass | 43.1 | 4 | — |
| 60 / 3 | 28555 | eagle3 / existing | pass | 47.9 | 4 | — |
| 61 / 0 | 28556 | eagle3 / fit_reduced_taps | pass | 103.6 | 8 | relay_reduced: 64.8% |
| 61 / 1 | 28556 | eagle3 / fit_reduced_taps | pass | 90.9 | 8 | relay_reduced: 95.5% |
| 61 / 2 | 28556 | eagle3 / fit_reduced_taps | pass | 177.7 | 8 | relay_reduced: 96.8% |
| 61 / 3 | 28556 | eagle3 / fit_reduced_taps | pass | 177.7 | 8 | relay_reduced: 98.2% |
| 62 / 0 | 28557 | dflash / existing | pass | 45.7 | 4 | — |
| 62 / 1 | 28557 | dflash / existing | pass | 46.6 | 4 | — |
| 62 / 2 | 28557 | eagle3 / existing | pass | 52.8 | 4 | — |
| 62 / 3 | 28557 | eagle3 / existing | pass | 49.0 | 4 | — |
| 63 / 0 | 28558 | eagle3 / native_columns | pass | 172.3 | 8 | relay_cropped: 81.6%; relay_inherited_u128: 96.4%; relay_inherited_u8192: 100.3%; relay_random_u128: 65.5%; relay_random_u8192: 98.3%; relay_svd1536: 88.5% |
| 63 / 1 | 28558 | eagle3 / native_columns | pass | 194.0 | 8 | relay_cropped: 78.8%; relay_inherited_u128: 95.3%; relay_inherited_u8192: 99.6%; relay_random_u128: 70.0%; relay_random_u8192: 95.7%; relay_svd1536: 92.9% |
| 63 / 2 | 28558 | eagle3 / native_columns | pass | 228.2 | 8 | relay_cropped: 82.9%; relay_inherited_u128: 95.3%; relay_inherited_u8192: 93.5%; relay_random_u128: 45.8%; relay_random_u8192: 92.5%; relay_svd1536: 93.5% |
| 63 / 3 | 28558 | eagle3 / native_columns | pass | 414.0 | 16 | relay_cropped: 82.7%; relay_inherited_u128: 93.7%; relay_inherited_u8192: 95.7%; relay_random_u128: 57.4%; relay_random_u8192: 95.2%; relay_svd1536: 87.8% |
| 64 / 0 | 28559 | dflash / existing | pass | 129.0 | 8 | — |
| 64 / 1 | 28559 | dflash / existing | pass | 128.7 | 8 | — |
| 64 / 2 | 28559 | eagle3 / existing | pass | 142.3 | 8 | — |
| 64 / 3 | 28559 | eagle3 / existing | pass | 149.8 | 8 | — |
| 65 / 0 | 28560 | eagle3 / existing | pass | 294.5 | 16 | relay_short: 99.9%; relay_svd1536: 92.9%; relay_two: 98.5% |
| 65 / 1 | 28560 | eagle3 / existing | pass | 281.5 | 16 | relay_short: 95.6%; relay_svd1536: 92.3%; relay_two: 99.1% |
| 65 / 2 | 28560 | eagle3 / existing | pass | 249.2 | 16 | relay_short: 98.2%; relay_svd1536: 90.9%; relay_two: 98.7% |
| 65 / 3 | 28560 | eagle3 / existing | pass | 248.6 | 16 | relay_short: 95.2%; relay_svd1536: 93.1%; relay_two: 98.1% |
| 66 / 0 | 28561 | dflash / existing | pass | 100.6 | 8 | — |
| 66 / 1 | 28561 | dflash / existing | pass | 101.7 | 8 | — |
| 66 / 2 | 28561 | eagle3 / existing | pass | 117.7 | 8 | — |
| 66 / 3 | 28561 | eagle3 / existing | pass | 109.9 | 8 | — |
| 67 / 0 | 28562 | dflash / existing | pass | 110.6 | 8 | — |
| 67 / 1 | 28562 | dflash / existing | pass | 113.3 | 8 | — |
| 67 / 2 | 28562 | eagle3 / existing | pass | 122.9 | 8 | — |
| 67 / 3 | 28562 | eagle3 / existing | pass | 130.0 | 8 | — |
| 68 / 0 | 28563 | dflash / existing | pass | 119.6 | 8 | — |
| 68 / 1 | 28563 | dflash / existing | pass | 121.9 | 8 | — |
| 68 / 2 | 28563 | eagle3 / existing | pass | 140.4 | 8 | — |
| 68 / 3 | 28563 | eagle3 / existing | pass | 131.7 | 8 | — |
| 69 / 0 | 28564 | dflash / existing | pass | 111.2 | 8 | — |
| 69 / 1 | 28564 | dflash / existing | pass | 109.3 | 8 | — |
| 69 / 2 | 28564 | eagle3 / existing | pass | 127.7 | 8 | — |
| 69 / 3 | 28564 | eagle3 / existing | pass | 119.7 | 8 | — |
| 70 / 0 | 28565 | dflash / existing | pass | 99.2 | 8 | — |
| 70 / 1 | 28565 | dflash / existing | pass | 103.1 | 8 | — |
| 70 / 2 | 28565 | eagle3 / existing | pass | 112.8 | 8 | — |
| 70 / 3 | 28565 | eagle3 / existing | pass | 112.9 | 8 | — |
| 71 / 0 | 28566 | dflash / existing | pass | 106.5 | 8 | — |
| 71 / 1 | 28566 | dflash / existing | pass | 113.3 | 8 | — |
| 71 / 2 | 28566 | eagle3 / existing | pass | 123.4 | 8 | — |
| 71 / 3 | 28566 | eagle3 / existing | pass | 127.9 | 8 | — |
| 72 / 0 | 28567 | dflash / existing | pass | 112.8 | 8 | — |
| 72 / 1 | 28567 | dflash / existing | pass | 127.6 | 8 | — |
| 72 / 2 | 28567 | eagle3 / existing | pass | 131.7 | 8 | — |
| 72 / 3 | 28567 | eagle3 / existing | pass | 145.0 | 8 | — |
| 73 / 0 | 28568 | dflash / existing | pass | 106.9 | 8 | — |
| 73 / 1 | 28568 | dflash / existing | pass | 114.1 | 8 | — |
| 73 / 2 | 28568 | eagle3 / existing | pass | 117.9 | 8 | — |
| 73 / 3 | 28568 | eagle3 / existing | pass | 124.0 | 8 | — |
| 74 / 0 | 28569 | eagle3 / fit_reduced_taps | pass | 117.2 | 8 | relay_reduced: 95.3%; relay_two_control: 95.3% |
| 74 / 1 | 28569 | eagle3 / fit_reduced_taps | pass | 113.7 | 8 | relay_reduced: 98.0%; relay_two_control: 95.1% |
| 74 / 2 | 28569 | eagle3 / fit_reduced_taps | pass | 114.4 | 8 | relay_reduced: 96.6%; relay_two_control: 95.2% |
| 74 / 3 | 28569 | eagle3 / fit_reduced_taps | pass | 117.8 | 8 | relay_reduced: 97.2%; relay_two_control: 95.4% |

## Four-GPU pipeline stages

These use stage-specific completion gates, not independent lane summaries.

| Wave | Job | Stage | Status | Elapsed | Gate |
|---:|---:|---|---|---|---|
| 20 | 28440 | boundary_pipeline_pilot | pass | 00:01:35 | pilot-gate.json |
| 21 | 28448 | boundary_extraction | pass | 00:00:42 | extraction-complete.json |
| 22 | 28455 | boundary_capacity_pilot | pass | 00:01:25 | batch-gate.json |
| 23 | 28457 | boundary_full_fit | pass | 00:01:20 | batch-gate.json |
| 24 | 28458 | boundary_evaluation | pass | 00:01:30 | campaign-gate.json |
