# Frozen DFlash 4B proposer to Qwen3-14B result

The selected coefficient-free relay passed the full exactly-four-GPU run and
both official scoring jobs.

| Population | Source tok/s | Relay tok/s | Relay/source | 95% CI | Accuracy S/R | Exact |
|---|---:|---:|---:|---:|---:|---:|
| MATH-500 all 500 | 110.450 | 137.604 | **1.2458x** | [1.2382, 1.2533] | 79.6/79.6 | 500/500 |
| Preregistered complement 468 | 110.644 | 137.727 | **1.2448x** | [1.2368, 1.2525] | 79.274/79.274 | 468/468 |

The full run emits 387,725 tokens per method. Both methods hit the declared
2,048-token cap on 9.4% of prompts. The cap is therefore reported as a fixed
comparability budget and its hit rate is exposed; it is not described as
truncation-free.

The earlier Part-1 table reported 1.311x against a 36-layer source path. The
new headline result uses the fairer optimized reference that stops after the
last proposer-consumed zero-based tap, layer 33. That raises source throughput
from 108.589 to 110.450 tok/s. The newly selected coefficient-free relay runs
at 137.604 tok/s; the old unmatched-training-history relay ran at 142.380
tok/s and remains only historical context, not the selected method.

## Mechanism evidence

- optimized source request time: 3,510.413 s;
- relay request time: 2,817.695 s;
- removable source-trunk share: 30.2%;
- relay plus relay-prefill share relative to source reference: 0.5%;
- micro committed tokens per cycle: 7.358 source versus 6.438 relay, or
  87.5% retention;
- measured Amdahl break-even retention: 69.7%;
- same-run Amdahl reconstruction: 1.249x;
- equal-acceptance ceiling: 1.423x.

The synchronized DFlash path measures TTFT. Source versus relay is 65.19/44.18
ms at p50 and 92.53/63.85 ms at p95, corresponding to 1.476x and 1.449x.
End-to-end request latency is 5.139/4.129 s at p50 and 18.783/15.270 s at p95,
corresponding to 1.245x and 1.230x. The primary throughput ratio remains total
paired output tokens divided by total paired request time, not a mean of
per-request ratios.

The development forecast was 1.265x and the untouched 468-question result is
1.245x, a 1.6% relative difference. Its paired lower confidence bound remains
above one, and measured acceptance retention exceeds break-even by 17.8
percentage points.

For deployment context only, the earlier same-hardware native autoregressive
control ran at 26.671 tok/s, so this relay run is approximately 5.160x native
AR. This is a cross-run contextual ratio because the source/relay and native
controls were not timed in one process. No released target-specific DFlash-14B
checkpoint is used, so no 14B native-proposer ceiling is claimed.

## Artifacts

- raw/scored rows and GPU telemetry: `reports/final/dflash-14b-math500/`;
- full official summary: `benchmark-paper-summary.json`;
- 468-question summary: `math500-confirmatory-paper-summary.json`;
- Amdahl profile: `analysis.json` and `analysis.md`;
- objective selection and checkpoint hash:
  `reports/design-selection/dflash/OBJECTIVE_SELECTION.md`.
