# Qwen3-14B two-turn MT-Bench performance

Job `25443` completed all 80 conversations / 160 turns in 32m33s using
exactly four RTX 6000 Ada GPUs. Each rank produced exactly 120 method rows,
for 480 total. Qwen3 was greedy and non-thinking with a 2,048-token cap.

| Method | End-to-end tok/s | Speedup vs source reuse | Mean accepted tokens/cycle | p95 TTFT | Cap hits |
|---|---:|---:|---:|---:|---:|
| Native AR | 26.5 | 0.601x [0.563, 0.638] | 1.00 | 347 ms | 1/160 |
| Qwen3-4B source reuse | 44.2 | 1.000x | 3.99 | 462 ms | 0/160 |
| RelaySpec | 51.0 | **1.154x [1.135, 1.176]** | 3.18 | 342 ms | 0/160 |

RelaySpec and source reuse are token-identical on all 160 turns. Their answers
average 502 output tokens (median 405), so this is a long-form chat regime and
not a short-output latency probe. The relay removes enough source-trunk work to
remain beneficial despite retaining 79.7% of source-reuse acceptance.

Per-cycle acceptance lists are present in every raw row, and the official
aggregate includes acceptance-survival probabilities by proposed position.
Speed intervals are 10,000-sample prompt-paired bootstrap 95% intervals.
