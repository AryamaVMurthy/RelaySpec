# Optimized source-reuse profile

This artifact is generated from immutable per-request rows. The reference is
`source_reuse_eagle3` and the candidate is `relay_eagle3`.

| Quantity | Result |
|---|---:|
| Paired requests | 468 |
| Exact sequence agreement | 464/468 (99.1%) |
| Reference request time | 4375.897 s |
| Relay request time | 3502.239 s |
| Observed end-to-end speedup | **1.249x** |
| Paired request-bootstrap 95% interval | [1.243, 1.255]x |
| Source-trunk runtime share | 33.1% |
| Relay runtime relative to reference | 0.4% |
| Acceptance retention | 84.0% |
| Amdahl-modeled speedup | 1.253x |
| Equal-acceptance ceiling | 1.485x |
| Break-even acceptance retention | 66.7% |

The bootstrap resamples whole paired requests, so prompt length and output-length
variation remain coupled across methods. The same-run Amdahl value reconstructs
latency from measured source, target/draft cycle, other, relay, and
micro-averaged committed-token components; it is a mechanism diagnostic, not
an independent prediction. A genuine held-out forecast must freeze these
quantities on development data before measuring the test workload.
