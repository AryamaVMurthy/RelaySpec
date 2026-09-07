# Expanded family-transfer results

Completed fitting runs: 16 / 16 (12 initial, 4 adaptive follow-ups).
Observed checkpoint/control screens: 74; block trials: 24; final lanes: 4.

Screening uses two requests capped at 256 new tokens. It is selection evidence only.
Leading configurations and neighboring block sizes are compared on eight development requests capped at 512 tokens.
Final confirmation uses 16 reserved requests capped at 1,024 new tokens.
Reference is FP32 greedy AR with TF32 disabled; drafter and source interface are BF16.

| Pair | Old decode TPS | Selected decode TPS | Old request TPS | Selected request TPS | Exact outputs |
|---|---:|---:|---:|---:|---:|
| llama | 113.63 | 113.66 | 111.86 | 111.90 | 16/16 |
| cross | 58.85 | 67.84 | 58.03 | 66.71 | 16/16 |

Training uses supplied Numina solution text, not generated rollouts. The new 4k control, 8k and 16k fits share the sampling recipe and validation split. Historical 4k training used a different recipe.

All candidates and fitting curves are retained in results.json. Exact output agreement is not an answer-quality metric.

Request TPS includes prefill and generation, excluding model loading, prompt tokenization and final text decoding.
