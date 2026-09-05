# RelaySpec benchmark report

All speed intervals are prompt-paired 95% bootstrap intervals.

## Main results

| Run | Benchmark | Method | Accuracy | Δ accuracy vs source | End-to-end tok/s | End-to-end speedup vs source | 95% CI | Acceptance | Cap rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| D14 | math500 | naive_source_reuse | 0.796 | 0.000 | 108.6 | 1.000 | [1.000, 1.000] | 7.606 | 0.094 |
| D14 | math500 | native_ar | 0.792 | -0.004 | 26.7 | 0.246 | [0.240, 0.251] | 1.000 | 0.088 |
| D14 | math500 | relay_p | 0.796 | 0.000 | 142.4 | 1.311 | [1.304, 1.318] | 6.805 | 0.094 |
| D8 | math500 | naive_source_reuse | 0.742 | 0.000 | 145.5 | 1.000 | [1.000, 1.000] | 7.760 | 0.102 |
| D8 | math500 | native_ar | 0.748 | 0.006 | 44.5 | 0.306 | [0.299, 0.313] | 1.000 | 0.120 |
| D8 | math500 | native_target_dflash | 0.742 | 0.000 | 241.2 | 1.658 | [1.650, 1.667] | 8.014 | 0.102 |
| D8 | math500 | relay_f | 0.742 | 0.000 | 218.6 | 1.503 | [1.494, 1.511] | 6.992 | 0.102 |

## Amdahl and component evidence

| Run | Method | Source-trunk wall share | Relay wall share | Profile accounted |
|---|---|---:|---:|---:|
| D14 | naive_source_reuse | 0.301 | 0.000 | 1.000 |
| D14 | native_ar | 0.000 | 0.000 | 1.000 |
| D14 | relay_p | 0.000 | 0.006 | 1.000 |
| D8 | naive_source_reuse | 0.386 | 0.000 | 1.000 |
| D8 | native_ar | 0.000 | 0.000 | 1.000 |
| D8 | native_target_dflash | 0.000 | 0.000 | 1.000 |
| D8 | relay_f | 0.000 | 0.008 | 1.000 |

## Reproducibility

- `D14`: `/home/aryamavmurthy/work/Latent-Recurrent/reports/d14-math500`
  - target: `Qwen/Qwen3-14B` @ `40c069824f4251a91eefaf281ebe4c544efd3e18`
  - scorer: `qwen2.5_math_official`
  - source hashes: `/home/aryamavmurthy/work/Latent-Recurrent/reports/d14-math500/source-sha256.txt`
- `D8`: `/home/aryamavmurthy/work/Latent-Recurrent/reports/d8-feature-math500`
  - target: `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218`
  - scorer: `qwen2.5_math_official`
  - source hashes: `/home/aryamavmurthy/work/Latent-Recurrent/reports/d8-feature-math500/source-sha256.txt`
