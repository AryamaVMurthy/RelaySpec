# Optimized source-reuse profile

This artifact is generated from immutable per-request rows. The reference is
`naive_source_reuse` and the candidate is `relay_f`.

| Quantity | Result |
|---|---:|
| Paired requests | 32 |
| Exact sequence agreement | 32/32 (100.0%) |
| Reference request time | 161.466 s |
| Relay request time | 108.621 s |
| Observed end-to-end speedup | **1.487x** |
| Paired request-bootstrap 95% interval | [1.454, 1.520]x |
| Source-trunk runtime share | 38.8% |
| Relay runtime relative to reference | 0.5% |
| Acceptance retention | 91.5% |
| Amdahl-modeled speedup | 1.486x |
| Equal-acceptance ceiling | 1.619x |
| Break-even acceptance retention | 61.0% |

The bootstrap resamples whole paired requests, so prompt length and output-length
variation remain coupled across methods. The same-run Amdahl value reconstructs
latency from measured source, target/draft cycle, other, relay, and
micro-averaged committed-token components; it is a mechanism diagnostic, not
an independent prediction. A genuine held-out forecast must freeze these
quantities on development data before measuring the test workload.
