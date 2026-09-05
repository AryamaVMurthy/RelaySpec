# RelaySpec on the pinned DeepSpec EAGLE-3 chain evaluator

Primary numbers below use the preregistered 468-problem complement that removes
all 32 architecture-development prompts by normalized hash. The conventional
full MATH-500 values remain in each run directory for comparability.

| Target | Source tok/s | Relay tok/s | Relay/source (95% paired CI) | Source share | Relay share | Acceptance retained | Official accuracy S/R | Byte agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | 83.05 | 103.77 | **1.250x [1.244, 1.256]** | 33.1% | 0.4% | 84.0% | 72.863% / 72.863% | 464/468 |
| Qwen3-14B | 64.93 | 70.20 | **1.081x [1.076, 1.087]** | 27.1% | 0.4% | 79.1% | 79.487% / 79.487% | 466/468 |

The frozen development forecasts were 1.238x at 8B and 1.090x at 14B. Held-out
throughput differs by +0.9% and -0.8%, respectively. Same-run Amdahl
reconstructions are 1.253x and 1.086x; these are labeled reconstructions rather
than independent forecasts.

Every finite-precision string mismatch is retained in the corresponding
`MISMATCH_AUDIT-confirmatory468.md`. All mismatches preserve the independently
scored task outcome, so the paired official accuracy delta and its bootstrap
interval are exactly zero at both scales.

These results use the released DeepSpec `ttt7` autoregressive-chain evaluator,
not a dynamic EAGLE tree. Native-AR and native-target-chain controls run in
separate memory-safe paired jobs. On full MATH-500, the native target-specific
chain reaches 114.28 tok/s at 8B and 78.78 tok/s at 14B. RelaySpec reaches
90.46% and 89.20% of those target-specific ceilings, respectively, while
matching every one of their official correctness outcomes. Detailed artifacts
are `EAGLE3_8B_NATIVE_CONTROL.md` and `EAGLE3_14B_NATIVE_CONTROL.md`.
