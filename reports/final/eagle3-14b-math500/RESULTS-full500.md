# Optimized source-reuse profile

This artifact is generated from immutable per-request rows. The reference is
`source_reuse_eagle3` and the candidate is `relay_eagle3`.

| Quantity | Result |
|---|---:|
| Paired requests | 500 |
| Exact sequence agreement | 498/500 (99.6%) |
| Reference request time | 5896.093 s |
| Relay request time | 5451.548 s |
| Observed end-to-end speedup | **1.082x** |
| Paired request-bootstrap 95% interval | [1.076, 1.087]x |
| Source-trunk runtime share | 27.1% |
| Relay runtime relative to reference | 0.4% |
| Acceptance retention | 79.1% |
| Amdahl-modeled speedup | 1.086x |
| Equal-acceptance ceiling | 1.365x |
| Break-even acceptance retention | 72.7% |

The bootstrap resamples whole paired requests, so prompt length and output-length
variation remain coupled across methods. The same-run Amdahl value reconstructs
latency from measured source, target/draft cycle, other, relay, and
micro-averaged committed-token components; it is a mechanism diagnostic, not
an independent prediction. A genuine held-out forecast must freeze these
quantities on development data before measuring the test workload.
