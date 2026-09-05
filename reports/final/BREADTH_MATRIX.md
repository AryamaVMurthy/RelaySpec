# RelaySpec frozen cross-task breadth matrix

| Family | Target | Task | N/clusters | Source tok/s | Relay tok/s | Speedup [95% CI] | Exact | Acceptance retained |
|---|---|---|---:|---:|---:|---:|---:|---:|
| eagle3 | 8b | gsm8k | 128/128 | 85.987 | 108.453 | 1.2613x [1.2437, 1.2789] | 99.22% | 84.31% |
| eagle3 | 8b | humaneval | 164/164 | 71.747 | 81.099 | 1.1304x [1.1172, 1.1434] | 96.95% | 75.18% |
| eagle3 | 8b | mbpp | 378/378 | 69.855 | 82.103 | 1.1753x [1.1656, 1.1850] | 97.62% | 77.84% |
| eagle3 | 8b | mtbench | 160/80 | 42.273 | 49.451 | 1.1698x [1.1501, 1.1906] | 89.38% | 80.87% |
| eagle3 | 14b | gsm8k | 128/128 | 67.807 | 73.407 | 1.0826x [1.0675, 1.0979] | 100.00% | 79.40% |
| eagle3 | 14b | humaneval | 164/164 | 55.384 | 50.577 | 0.9132x [0.9011, 0.9255] | 99.39% | 66.27% |
| eagle3 | 14b | mbpp | 378/378 | 54.549 | 50.492 | 0.9256x [0.9151, 0.9361] | 98.68% | 66.77% |
| eagle3 | 14b | mtbench | 160/80 | 33.828 | 34.575 | 1.0221x [0.9999, 1.0449] | 96.88% | 76.37% |
| dflash | 8b | gsm8k | 128/128 | 116.869 | 166.388 | 1.4237x [1.4040, 1.4445] | 100.00% | 85.36% |
| dflash | 8b | humaneval | 164/164 | 118.844 | 144.999 | 1.2201x [1.2017, 1.2383] | 100.00% | 72.91% |
| dflash | 8b | mbpp | 378/378 | 117.772 | 151.554 | 1.2868x [1.2732, 1.3001] | 100.00% | 76.46% |
| dflash | 8b | mtbench | 160/80 | 60.851 | 81.616 | 1.3412x [1.3183, 1.3639] | 100.00% | 82.90% |
| dflash | 14b | gsm8k | 128/128 | 88.045 | 97.072 | 1.1025x [1.0836, 1.1214] | 100.00% | 76.30% |
| dflash | 14b | humaneval | 164/164 | 87.337 | 77.609 | 0.8886x [0.8741, 0.9038] | 100.00% | 61.48% |
| dflash | 14b | mbpp | 378/378 | 88.020 | 83.977 | 0.9541x [0.9416, 0.9663] | 100.00% | 65.28% |
| dflash | 14b | mtbench | 160/80 | 45.319 | 49.295 | 1.0877x [1.0653, 1.1083] | 100.00% | 76.01% |

Both MT-Bench turns are one bootstrap cluster. Official code quality is stored in the JSON artifact; MT-Bench has no quality claim without a judge run.

## Cell summaries

| Family | Target | Geometric-mean speedup | Positive 95% CI cells | Amdahl direction matches | Amdahl mean relative error |
|---|---|---:|---:|---:|---:|
| eagle3 | 8b | 1.1832x | 4/4 | 4/4 | 0.35% |
| eagle3 | 14b | 0.9834x | 1/4 | 4/4 | 0.61% |
| dflash | 8b | 1.3159x | 4/4 | 4/4 | 0.03% |
| dflash | 14b | 1.0042x | 2/4 | 4/4 | 0.36% |

## Complete matrix summary

The geometric means give every family/target/task cell equal weight; they are descriptive and do not assume an operational workload mixture.

- Raw RelaySpec geometric-mean speedup: 1.1135x.
- Cells with paired 95% speed interval above one: 11/16.
- Amdahl direction matches: 16/16.
- Amdahl mean absolute relative error: 0.34%.
- Descriptive profile-policy geometric-mean speedup: 1.1354x.

## Automatic coefficient-free profile policy

The policy selects RelaySpec only when the measured Amdahl prediction and the paired 95% speed lower bound both exceed the source boundary of one; otherwise it retains source reuse. No quality proxy or tuned threshold enters the decision.

| Family | Target | Task | Predicted | Selected provider | Realized speedup | Direction match |
|---|---|---|---:|---|---:|---:|
| eagle3 | 8b | gsm8k | 1.2644x | relay_eagle3 | 1.2613x | yes |
| eagle3 | 8b | humaneval | 1.1345x | relay_eagle3 | 1.1304x | yes |
| eagle3 | 8b | mbpp | 1.1799x | relay_eagle3 | 1.1753x | yes |
| eagle3 | 8b | mtbench | 1.1745x | relay_eagle3 | 1.1698x | yes |
| eagle3 | 14b | gsm8k | 1.0872x | relay_eagle3 | 1.0826x | yes |
| eagle3 | 14b | humaneval | 0.9194x | source_reuse_eagle3 | 1.0000x | yes |
| eagle3 | 14b | mbpp | 0.9320x | source_reuse_eagle3 | 1.0000x | yes |
| eagle3 | 14b | mtbench | 1.0286x | source_reuse_eagle3 | 1.0000x | yes |
| dflash | 8b | gsm8k | 1.4228x | relay_p | 1.4237x | yes |
| dflash | 8b | humaneval | 1.2203x | relay_p | 1.2201x | yes |
| dflash | 8b | mbpp | 1.2866x | relay_p | 1.2868x | yes |
| dflash | 8b | mtbench | 1.3408x | relay_p | 1.3412x | yes |
| dflash | 14b | gsm8k | 1.1063x | relay_p | 1.1025x | yes |
| dflash | 14b | humaneval | 0.8926x | optimized_source_reuse | 1.0000x | yes |
| dflash | 14b | mbpp | 0.9567x | optimized_source_reuse | 1.0000x | yes |
| dflash | 14b | mtbench | 1.0917x | relay_p | 1.0877x | yes |

## Measured Amdahl mechanism check

| Family | Target | Task | Source share | Relay/reference | Retention | Break-even | Predicted | Observed | Equal-acceptance ceiling |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| eagle3 | 8b | gsm8k | 33.69% | 0.38% | 83.84% | 65.88% | 1.2644x | 1.2613x | 1.4994x |
| eagle3 | 8b | humaneval | 33.46% | 0.42% | 75.36% | 66.23% | 1.1345x | 1.1304x | 1.4935x |
| eagle3 | 8b | mbpp | 33.56% | 0.41% | 78.28% | 66.06% | 1.1799x | 1.1753x | 1.4960x |
| eagle3 | 8b | mtbench | 33.20% | 0.39% | 78.37% | 66.46% | 1.1745x | 1.1698x | 1.4883x |
| eagle3 | 14b | gsm8k | 27.36% | 0.41% | 78.80% | 72.31% | 1.0872x | 1.0826x | 1.3691x |
| eagle3 | 14b | humaneval | 27.23% | 0.47% | 66.62% | 72.60% | 0.9194x | 0.9132x | 1.3653x |
| eagle3 | 14b | mbpp | 27.26% | 0.47% | 67.44% | 72.49% | 0.9320x | 0.9256x | 1.3659x |
| eagle3 | 14b | mtbench | 26.83% | 0.41% | 75.08% | 72.94% | 1.0286x | 1.0221x | 1.3591x |
| dflash | 8b | gsm8k | 40.00% | 0.56% | 85.68% | 59.60% | 1.4228x | 1.4237x | 1.6513x |
| dflash | 8b | humaneval | 39.46% | 0.64% | 73.96% | 60.31% | 1.2203x | 1.2201x | 1.6344x |
| dflash | 8b | mbpp | 39.57% | 0.61% | 77.84% | 60.07% | 1.2866x | 1.2868x | 1.6384x |
| dflash | 8b | mtbench | 38.52% | 0.56% | 82.73% | 61.27% | 1.3408x | 1.3412x | 1.6119x |
| dflash | 14b | gsm8k | 30.92% | 0.59% | 76.39% | 68.85% | 1.1063x | 1.1025x | 1.4353x |
| dflash | 14b | humaneval | 30.46% | 0.72% | 61.88% | 69.52% | 0.8926x | 0.8886x | 1.4233x |
| dflash | 14b | mbpp | 30.59% | 0.67% | 66.10% | 69.18% | 0.9567x | 0.9541x | 1.4271x |
| dflash | 14b | mtbench | 29.80% | 0.56% | 76.69% | 70.10% | 1.0917x | 1.0877x | 1.4131x |
