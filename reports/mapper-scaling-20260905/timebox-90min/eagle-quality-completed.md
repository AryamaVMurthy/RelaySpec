# Completed EAGLE 8B quality evaluation

Audited 128 exposed development questions, 11 methods, 1,408 rows; output cap 2,048 tokens. Both shards completed successfully. Exact coverage, checkpoint/source provenance, pilot prerequisite and pinned scorer replay passed.

| Method | Tokens/s | Dense N2048 retention (95% paired CI) | Correct / 128 | Cap outputs |
|---|---:|---:|---:|---:|
| native_ar | 36.32 | 38.19% [37.14, 39.25] | 104 | 11 |
| native_target_eagle3 | 94.89 | 99.78% [98.86, 100.80] | 105 | 7 |
| relay_eagle3_dense_n2048 | 95.10 | 100.00% [100.00, 100.00] | 105 | 7 |
| relay_eagle3_dense_n512 | 93.47 | 98.28% [97.59, 99.01] | 105 | 7 |
| relay_eagle3_factorized2048_n2048 | 93.59 | 98.41% [97.76, 99.12] | 105 | 7 |
| relay_eagle3_factorized2048_n512 | 93.85 | 98.69% [98.08, 99.31] | 105 | 7 |
| relay_eagle3_factorized512_n2048 | 84.91 | 89.29% [88.54, 90.09] | 105 | 7 |
| relay_eagle3_mlp2048_n2048 | 89.25 | 93.85% [93.27, 94.46] | 105 | 7 |
| relay_eagle3_mlp2048_n512 | 84.67 | 89.03% [88.24, 89.87] | 105 | 7 |
| relay_eagle3_mlp512_n2048 | 79.76 | 83.87% [82.94, 84.84] | 105 | 7 |
| source_reuse_eagle3 | 66.80 | 70.24% [69.69, 70.84] | 105 | 7 |

Dense N512 retains 98.28% of dense N2048 throughput (95% paired CI 97.59–99.01%). Linear width2048 N512 retains 98.69%; neither tested MLP2048 endpoint exceeds dense throughput. These support small-data efficiency and diminishing capacity returns in a second drafter family.

All speculative methods have the same per-question correctness as dense N2048 in this sample. The conservative paired 95% accuracy-difference interval is still ±3.37 percentage points; equal counts do not prove 1-point non-inferiority. These are exposed development questions and capped generations, not untouched confirmation or uncapped answer-quality evidence.

The paper rewrite remains deferred until the 90-minute experiment window closes. Paper assets were replayed/exported successfully under /tmp only; the manuscript has not been updated with this result yet.

The targeted rate pilot began after the second shard. At 06:02:16 UTC fewer than ten minutes remained before the 06:11:48 UTC GPU cutoff, so no proper ten-minute follow-up fit or decoding job can be launched under the declared rule. Do not promote 16-update pilot fits to endpoint scientific results.
