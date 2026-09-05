# Optimized source-reuse profile

This artifact is generated from immutable per-request rows. The reference is
`source_reuse_eagle3` and the candidate is `relay_eagle3`.

| Quantity | Result |
|---|---:|
| Paired requests | 32 |
| Exact sequence agreement | 32/32 (100.0%) |
| Reference request time | 305.947 s |
| Relay request time | 247.382 s |
| Observed end-to-end speedup | **1.237x** |
| Paired request-bootstrap 95% interval | [1.213, 1.263]x |
| Source-trunk runtime share | 33.3% |
| Relay runtime relative to reference | 0.4% |
| Acceptance retention | 82.6% |
| Amdahl-modeled speedup | 1.238x |
| Equal-acceptance ceiling | 1.491x |
| Break-even acceptance retention | 66.4% |

The bootstrap resamples whole paired requests, so prompt length and output-length
variation remain coupled across methods. The same-run Amdahl value reconstructs
latency from measured source, target/draft cycle, other, relay, and
micro-averaged committed-token components; it is a mechanism diagnostic, not
an independent prediction. A genuine held-out forecast must freeze these
quantities on development data before measuring the test workload.
