# EAGLE-3 Qwen3-14B native control

All values below come from immutable exactly-four-GPU jobs. Source/relay and
native-AR/native-target pairs run in separate memory-safe processes; the former
is the paired causal comparison and cross-group ratios are matched-protocol
deployment-frontier comparisons.

## Full MATH-500

| Method | End-to-end tok/s | Official accuracy | Cap rate |
|---|---:|---:|---:|
| Native Qwen3-14B AR | 26.712 | 79.2% | 8.8% |
| Native target-specific DeepSpec EAGLE-3 chain | 78.780 | 79.6% | 8.0% |
| Qwen3-4B source reuse | 64.969 | 79.6% | 8.0% |
| RelaySpec | 70.268 | 79.6% | 8.0% |

- Native target-specific EAGLE is 2.944x [2.900, 2.988] faster than native AR
  by output throughput.
- RelaySpec is 1.082x [1.076, 1.087] faster than optimized source reuse, 2.631x
  the native-AR throughput, and 89.20% of the target-specific EAGLE ceiling.
- RelaySpec matches every target-specific EAGLE correctness outcome and agrees
  with it on 499/500 output sequences.
- Native target-specific EAGLE has a +0.4 percentage-point accuracy delta
  versus native AR with paired 95% interval [-1.4, +2.2]. Native AR byte
  agreement is 23.4%; this finite-precision block-versus-token diagnostic is
  kept separate from the exact-arithmetic target-authority argument.

## Preregistered 468-question complement

| Method | End-to-end tok/s | Official accuracy | Cap rate |
|---|---:|---:|---:|
| Native Qwen3-14B AR | 26.713 | 78.419% | 8.97% |
| Native target-specific DeepSpec EAGLE-3 chain | 78.783 | 79.487% | 7.91% |
| Qwen3-4B source reuse | 64.928 | 79.487% | 7.91% |
| RelaySpec | 70.203 | 79.487% | 7.91% |

On the independent complement, native target-specific EAGLE is 2.944x
[2.899, 2.990] over native AR. RelaySpec is 1.081x [1.076, 1.087] over source
reuse, 2.628x native-AR throughput, and 89.11% of native target-specific EAGLE.
The native-target versus AR accuracy delta is +1.068 points with paired 95%
interval [-0.641, +2.991].

## Evidence locations

- Native controls: `reports/final/eagle3-14b-math500-native/`
- Matched source/relay: `reports/final/eagle3-14b-math500/`
- Full mismatch audit:
  `reports/final/eagle3-14b-math500-native/benchmark-mismatch-audit.json`
- Confirmatory mismatch audit:
  `reports/final/eagle3-14b-math500-native/math500-confirmatory-mismatch-audit.json`
