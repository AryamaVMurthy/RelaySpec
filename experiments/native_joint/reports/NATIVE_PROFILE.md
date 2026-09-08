# Native generation time attribution

Eight development requests, two per workload, at a512-token output cap. CUDA-event instrumentation exactly reproduced all native tokens and acceptance lengths. Event spans may include GPU idle gaps during host dispatch; instrumentation adds overhead. These are diagnostic fractions, not kernel-only timings or candidate speed measurements. Target vocabulary-head spans are subtracted from target verification before summing components.

| Component | Seconds across requests | Fraction of instrumented wall time |
|---|---:|---:|
| Target backbone | 15.856 | 75.8% |
| Target vocabulary head | 0.992 | 4.7% |
| Draft backbone | 2.582 | 12.3% |
| Draft vocabulary head | 0.991 | 4.7% |
| Prefill | 0.254 | 1.2% |
| Other / unassigned | 0.248 | 1.2% |

Using these diagnostic spans as an Amdahl-style estimate, removing all target vocabulary-head time would yield only about1.050× throughput if all other work stayed fixed. Lazy projection also adds synchronization. This supports deprioritizing head-only verification optimizations for the10% objective and focusing on accepted progress per expensive target pass.
