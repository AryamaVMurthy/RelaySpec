# Packed tree development results

These are adaptive development results. Additional excludes the initial eight pilot requests, but is not the reserved confirmation set. Request bootstrap intervals exclude search selection uncertainty and do not establish a10% lower bound. Timings include prefill; cap512 unless the variant explicitly states otherwise. See raw results for numerical output agreement.

| Configuration | Requests | Candidate / native TPS | Ratio [paired95% interval] |
|---|---:|---:|---|
| run-29315 tree4_b16_breadth (all) | 32 | 135.9 / 128.7 | 1.056 [1.015, 1.095] |
| run-29315 tree4_b16_breadth (additional) | 24 | 136.5 / 131.2 | 1.040 [1.003, 1.081] |
| run-29337 tree4_suffix8_breadth (all) | 32 | 135.2 / 128.5 | 1.052 [1.012, 1.088] |
| run-29337 tree4_suffix8_breadth (additional) | 24 | 135.9 / 131.1 | 1.037 [1.005, 1.069] |
| run-29338 leaf_top5_p4_breadth (all) | 32 | 144.0 / 128.6 | 1.120 [1.065, 1.168] |
| run-29338 leaf_top5_p4_breadth (additional) | 24 | 145.2 / 131.2 | 1.107 [1.053, 1.159] |
| run-29338 tree5_suffix6_breadth (all) | 32 | 141.1 / 128.9 | 1.095 [1.057, 1.132] |
| run-29338 tree5_suffix6_breadth (additional) | 24 | 142.0 / 131.4 | 1.081 [1.045, 1.120] |
| run-29381 ddtree47_32requests_2048 (all) | 32 | 155.1 / 127.9 | 1.213 [1.148, 1.270] |
| run-29381 ddtree47_32requests_2048 (additional) | 24 | 161.3 / 133.3 | 1.210 [1.141, 1.276] |
| run-29381 ddtree63_32requests_2048 (all) | 32 | 156.5 / 128.4 | 1.219 [1.156, 1.279] |
| run-29381 ddtree63_32requests_2048 (additional) | 24 | 163.4 / 133.8 | 1.221 [1.152, 1.290] |

Direct prior art: DDTree (Ringel and Romano, arXiv2604.12989). Our baseline adapts its pinned MIT heap builder into the shared runtime; no general tree novelty claim.

Selected four BF16 divergent cases matched exactly after FP32 parameter conversion of both models. All had packed-target top-logit ties. This is diagnostic evidence for numerical sensitivity in those cases, not universal equality or task-quality validation.
