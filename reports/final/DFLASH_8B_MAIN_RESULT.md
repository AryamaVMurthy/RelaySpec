# Frozen DFlash 4B proposer to Qwen3-8B result

The selected coefficient-free relay passed the full exactly-four-GPU run and
both official scoring jobs.

| Population | Source tok/s | Relay tok/s | Relay/source | 95% CI | Accuracy S/R | Exact |
|---|---:|---:|---:|---:|---:|---:|
| MATH-500 all 500 | 148.332 | 219.991 | **1.4831x** | [1.4745, 1.4917] | 74.2/74.2 | 500/500 |
| Preregistered complement 468 | 148.499 | 220.011 | **1.4816x** | [1.4726, 1.4902] | 73.932/73.932 | 468/468 |

The full run emits 390,145 tokens per method. Both methods hit the declared
2,048-token cap on 10.2% of prompts, so the cap is reported as a comparability
choice rather than described as truncation-free.

The earlier Part-1 table reported 1.503x against a 36-layer source path. The
new headline ratio is intentionally 1.483x because the causal reference now
executes only through the final required zero-based tap 33. This strengthens
the baseline from 145.478 to 148.332 tok/s while the newly selected relay also
improves from 218.642 to 219.991 tok/s. The smaller ratio is the fairer result,
not a regression hidden by denominator choice.

## Mechanism evidence

- optimized source request time: 2,630.207 s;
- relay request time: 1,773.456 s;
- removable source-trunk share: 38.8%;
- relay plus relay-prefill share relative to source reference: about 0.5%;
- micro acceptance: 7.761 source versus 7.065 relay, or 91.3% retention;
- measured Amdahl break-even retention: 61.0%;
- same-run Amdahl reconstruction: 1.483x;
- equal-acceptance ceiling: 1.620x.

The synchronized DFlash path also measures TTFT: source versus relay is
46.73/25.87 ms at p50 and 58.78/34.37 ms at p95, corresponding to source/relay
ratios of 1.806x and 1.710x. EAGLE TTFT remains unavailable because its inherited
backend emits a zero placeholder; that sentinel is never mixed into this
measured DFlash result.

End-to-end request latency is 3.832/2.525 s at p50 and 13.801/9.319 s at p95
for source/relay, giving ratios of 1.518x and 1.481x. These percentiles retain
per-request output-length variation; the primary aggregate ratio remains the
ratio of summed paired latency rather than a mean of per-request ratios.

The development forecast was 1.517x and the untouched 468-question result is
1.482x, a 2.4% relative difference. The final speed remains well above both
the no-slowdown boundary and its paired confidence criterion.

For deployment context only, immutable same-hardware controls from the earlier
full run give native AR at 44.516 tok/s and released target-specific DFlash-8B
at 241.245 tok/s. The frozen RelaySpec result is therefore approximately
4.942x native AR and retains 91.19% of the target-specific DFlash ceiling.
These contextual ratios are not substituted for the causal same-run
source-versus-relay result because their timing comes from a separate run.
The corresponding native-AR accuracy is 74.8%; target-specific DFlash and the
historical relay score 74.2%, a paired block-versus-token-kernel delta of
-0.6 percentage points with interval [-2.8, +1.4]. This is reported separately
from the exactly zero source-versus-relay delta.

## Artifacts

- raw/scored rows and GPU telemetry: `reports/final/dflash-8b-math500/`;
- full official summary: `benchmark-paper-summary.json`;
- 468-question summary: `math500-confirmatory-paper-summary.json`;
- Amdahl profile: `analysis.json` and `analysis.md`;
- objective selection and checkpoint hash:
  `reports/design-selection/dflash/OBJECTIVE_SELECTION.md`.
