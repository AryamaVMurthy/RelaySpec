# Optimized source-reuse profile

This artifact is generated from immutable per-request rows. The reference is
`optimized_source_reuse` and the candidate is `relay_p`.

| Quantity | Result |
|---|---:|
| Paired requests | 32 |
| Exact sequence agreement | 32/32 (100.0%) |
| Reference request time | 227.222 s |
| Relay request time | 181.571 s |
| Observed end-to-end speedup | **1.251x** |
| Paired request-bootstrap 95% interval | [1.229, 1.276]x |
| Source-trunk runtime share | 30.2% |
| Relay runtime relative to reference | 0.5% |
| Acceptance retention | 88.1% |
| Amdahl-modeled speedup | 1.256x |
| Equal-acceptance ceiling | 1.422x |
| Break-even acceptance retention | 69.7% |

The bootstrap resamples whole paired requests, so prompt length and output-length
variation remain coupled across methods. The same-run Amdahl value reconstructs
latency from measured source, target/draft cycle, other, relay, and
micro-averaged committed-token components; it is a mechanism diagnostic, not
an independent prediction. A genuine held-out forecast must freeze these
quantities on development data before measuring the test workload.
