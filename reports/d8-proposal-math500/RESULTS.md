# Qwen3-8B aggressive proposal-aligned ablation

Jobs `25435`/`25436` evaluate the feature relay after 256 warm-started steps
with proposal KL weight 1.0 and learning rate 5e-5. This is an objective
ablation, not the selected RelaySpec model.

| Method | Accuracy | End-to-end tok/s | Mean accepted | Speedup vs source reuse |
|---|---:|---:|---:|---:|
| Native AR | 74.8% | 44.5 | 1.00 | 0.305x |
| Naive 4B source reuse | 74.2% | 145.8 | 7.76 | 1.000x |
| Official target-specific DFlash-8B | 74.2% | 241.3 | 8.01 | 1.654x |
| Aggressive proposal-aligned relay | 74.2% | 179.8 | 5.73 | 1.233x |

The relay remains exactly output-preserving relative to source reuse on all
500/500 prompts, with equal 74.2% official accuracy. Its end-to-end speedup is
1.233x [1.223, 1.242], but it is substantially slower than the selected
feature relay's 1.503x [1.494, 1.511]. Training diagnostics show feature MSE
and cosine distance rising after the warm start: the optimizer moved out of
the useful feature solution and reduced acceptance. This establishes why
proposal alignment must be a trust-region-style correction rather than a
dominant objective. The queued conservative refinement tests that correction
without changing the selected result unless it wins the smoke gate.

Raw paired evidence contains four 500-row rank files under `raw/`.
