# 128-question evaluation, 2,048-token cap

Status: COMPLETE

| Pair | Mapper | Requests | Exact vs FP32 AR | Decode tokens/s | Request tokens/s |
|---|---|---:|---:|---:|---:|
| llama | ar | 128 | 128/128 | 45.89 | 45.73 |
| llama | old | 128 | 128/128 | 116.00 | 114.93 |
| llama | selected | 128 | 128/128 | 116.52 | 115.44 |
| cross | ar | 128 | 128/128 | 21.66 | 21.61 |
| cross | old | 128 | 128/128 | 52.29 | 51.97 |
| cross | selected | 128 | 128/128 | 59.83 | 59.41 |

Full sequence comparisons and first-divergence positions are in status.json. Raw token IDs, timing, and hashes are in artifacts/. See PLAN.md and protocol.json for selection and precision. These are token-sequence matches, not task-answer accuracy.
