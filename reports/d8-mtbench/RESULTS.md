# Qwen3-8B two-turn MT-Bench performance

Job `25445` completed all 80 conversations / 160 turns in 25m21s using
exactly four RTX 6000 Ada GPUs. Each rank produced exactly 160 method rows,
for 640 total. Qwen3 was greedy and non-thinking with a 2,048-token cap.

| Method | End-to-end tok/s | Speedup vs source reuse | Mean accepted tokens/cycle | p95 TTFT | Cap hits |
|---|---:|---:|---:|---:|---:|
| Native AR | 44.3 | 0.740x [0.696, 0.785] | 1.00 | 167 ms | 1/160 |
| Qwen3-4B source reuse | 59.8 | 1.000x | 4.03 | 303 ms | 1/160 |
| Official target-specific DFlash-8B | 100.5 | 1.680x [1.665, 1.696] | 4.25 | 183 ms | 1/160 |
| RelaySpec | 79.9 | **1.336x [1.316, 1.357]** | 3.30 | 183 ms | 1/160 |

RelaySpec, source reuse, and target-specific DFlash are token-identical on all
160 turns. Their answers average 505 output tokens (median 380). RelaySpec
reaches 79.5% of target-specific DFlash throughput without training or storing
a new target-specific proposer, and is 1.80x faster than native AR end to end.

Per-cycle acceptance lists are present in every raw row, and the official
aggregate includes acceptance-survival probabilities by proposed position.
Speed intervals are 10,000-sample prompt-paired bootstrap 95% intervals.
