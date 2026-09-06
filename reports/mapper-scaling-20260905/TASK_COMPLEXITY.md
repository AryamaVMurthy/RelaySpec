# Input-only task complexity analysis

Labels use original benchmark metadata and shared input tokens only. No candidate success, output length or timing enters assignment. Levels, subjects and input length are observational correlates, not causal interventions. Empty groups remain explicit. No additional generation, fitting, distinct examples or held-out/reserve access. 10000 paired request bootstrap samples for descriptive throughput only. Conservative exact paired intervals for accuracy differences. No multiplicity-adjusted confirmatory subgroup claim.

Every one of the128 requests and all12 methods is retained. Groups were fixed before computing subgroup outcomes. Labels come from MATH metadata and common tokenized input length. This is a retrospective development analysis, not a held-out experiment.

| Difficulty | Requests | Dense512 / dense2048 throughput [95% CI] | Linear1024 / dense2048 [95% CI] | MLP4096 / dense2048 [95% CI] | AR correct | Mapper correct |
| --- | ---: | --- | --- | --- | ---: | ---: |
| level_1_2 | 42 | 0.974 [0.961, 0.986] | 0.945 [0.932, 0.959] | 0.955 [0.943, 0.966] | 38 | 38 |
| level_3 | 21 | 0.979 [0.961, 1.002] | 0.948 [0.934, 0.965] | 0.968 [0.949, 0.987] | 19 | 18 |
| level_4_5 | 65 | 0.979 [0.970, 0.988] | 0.952 [0.940, 0.964] | 0.960 [0.951, 0.969] | 47 | 49 |

Dense512 retains97.4--97.9% of dense2048 throughput across the three difficulty groups. Linear1024 stays near the95% boundary, and MLP4096 retains95.4--96.8%. These descriptive subgroup intervals do not establish noninferiority or adjust for model selection and multiple comparisons.

The highest difficulty group contains all11 AR cap hits and8 of9 mapper cap hits. Cap hits are not necessarily truncations. Quality differences and model output lengths must be kept separate from speed interpretation.

Input-token groups contain52,57,15 and4 requests at<=64,65--128,129--256 and>256 tokens. In the final four-request group, dense512 retains94.5% and linear1024 retains90.8% of dense2048 throughput. This is a sparse observational signal requiring replication, not a basis for retuning the subgroup or asserting a causal length effect.

All seven original subjects and all methods, accuracy intervals, cap counts, input-only assignments and raw hashes are retained in task-complexity-results.json. Missing native-drafter accepted/proposed token counts are explicit nulls rather than zero acceptance. No generation, training or confirmation/reserve access was performed.
