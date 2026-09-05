# Optimized source-reuse profile

This artifact is generated from immutable per-request rows. The reference is
`optimized_source_reuse` and the candidate is `relay_p`.

| Quantity | Result |
|---|---:|
| Paired requests | 500 |
| Exact sequence agreement | 500/500 (100.0%) |
| Reference request time | 2630.207 s |
| Relay request time | 1773.456 s |
| Observed end-to-end speedup | **1.483x** |
| Paired request-bootstrap 95% interval | [1.475, 1.492]x |
| Source-trunk runtime share | 38.8% |
| Relay runtime relative to reference | 0.5% |
| Acceptance retention | 91.3% |
| Amdahl-modeled speedup | 1.483x |
| Equal-acceptance ceiling | 1.620x |
| Break-even acceptance retention | 61.0% |

The bootstrap resamples whole paired requests, so prompt length and output-length
variation remain coupled across methods. The same-run Amdahl value reconstructs
latency from measured source, target/draft cycle, other, relay, and
micro-averaged committed-token components; it is a mechanism diagnostic, not
an independent prediction. A genuine held-out forecast must freeze these
quantities on development data before measuring the test workload.
