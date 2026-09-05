# EAGLE-3 Qwen3-8B native control

All values below come from immutable exactly-four-GPU jobs. The source/relay
pair and the native-AR/native-target pair were deliberately loaded in separate
processes to avoid changing timings through incompatible model residency.
Therefore source versus relay is the paired causal comparison; ratios across
the two groups are matched-protocol deployment-frontier comparisons.

## Full MATH-500

| Method | End-to-end tok/s | Official accuracy | Cap rate |
|---|---:|---:|---:|
| Native Qwen3-8B AR | 44.447 | 74.8% | 12.0% |
| Native target-specific DeepSpec EAGLE-3 chain | 114.278 | 73.2% | 11.8% |
| Qwen3-4B source reuse | 82.827 | 73.2% | 11.8% |
| RelaySpec | 103.371 | 73.2% | 11.8% |

- Native target-specific EAGLE is 2.567x [2.534, 2.600] faster than native AR
  by output throughput.
- RelaySpec is 1.248x [1.242, 1.254] faster than optimized source reuse, 2.326x
  the native-AR throughput, and 90.46% of the target-specific EAGLE ceiling.
- RelaySpec/source official accuracy delta is exactly zero. RelaySpec also
  matches every one of the target-specific EAGLE correctness outcomes and
  agrees with it on 497/500 output sequences.
- Native target-specific EAGLE differs from one-token AR on 367/500 byte
  sequences and has a -1.6 percentage-point accuracy delta with paired 95%
  interval [-3.8, +0.4]. This is not attributed to RelaySpec: source reuse,
  RelaySpec, and target-specific EAGLE all use block-shaped verification and
  score 73.2%, while one-token AR uses different BF16 kernel shapes.

## Preregistered 468-question complement

| Method | End-to-end tok/s | Official accuracy | Cap rate |
|---|---:|---:|---:|
| Native Qwen3-8B AR | 44.452 | 74.573% | 11.54% |
| Native target-specific DeepSpec EAGLE-3 chain | 114.664 | 72.863% | 11.75% |
| Qwen3-4B source reuse | 83.048 | 72.863% | 11.75% |
| RelaySpec | 103.770 | 72.863% | 11.75% |

On the independent complement, native target-specific EAGLE is 2.575x
[2.542, 2.610] over native AR. RelaySpec is 1.250x [1.244, 1.256] over source
reuse, 2.334x native-AR throughput, and 90.50% of native target-specific EAGLE.
It agrees with native target-specific EAGLE on 465/468 sequences.

## Evidence locations

- Native controls: `reports/final/eagle3-8b-math500-native/`
- Matched source/relay: `reports/final/eagle3-8b-math500/`
- Full mismatch audit:
  `reports/final/eagle3-8b-math500-native/benchmark-mismatch-audit.json`
- Confirmatory mismatch audit:
  `reports/final/eagle3-8b-math500-native/math500-confirmatory-mismatch-audit.json`
