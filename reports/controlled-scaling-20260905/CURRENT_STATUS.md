# Experiment status and interpretation

Snapshot checked: 2026-09-05T15:24:28.074529+00:00.

Six of seven jobs in the controlled batch completed with Slurm exit code 0:0. Job 27583 is running on node07 with four L40S GPUs. At the live check it had run 44 minutes. Training and all five checkpoint saves have finished. The 128-update evaluation passed its 512-row gate and was scored locally. The 256-update evaluation is still finishing. The collector is active. Allow approximately 65–90 more minutes for remaining evaluations, then CPU collection and scoring. This estimate is based on observed 22-minute single-checkpoint jobs, not the four-hour job limit.

## Newly completed EAGLE-3 breadth

Use the released 4B drafter and selected fitted map with 8B or 14B targets. Run AR, source reuse and RelaySpec on the same workloads. Each target has 830 requests: 128 GSM8K, 164 HumanEval, 378 MBPP and 160 turns from 80 MT-Bench conversations. There are 2,490 method/request records per target. This tests workload coverage without repeating the main MATH-500 runs.

| Target | Workload | AR tokens/s | Relay tokens/s | AR speedup [95% CI] | Draft acceptance |
| --- | --- | ---: | ---: | --- | ---: |
| 14B | gsm8k | 22.97 | 61.40 | 2.673 [2.605, 2.744] | 44.53% |
| 14B | humaneval | 22.94 | 42.22 | 1.841 [1.805, 1.878] | 26.20% |
| 14B | mbpp | 22.93 | 41.89 | 1.826 [1.790, 1.868] | 25.85% |
| 14B | mtbench | 22.83 | 29.20 | 1.279 [1.204, 1.372] | 14.26% |
| 8B | gsm8k | 36.99 | 88.88 | 2.403 [2.348, 2.462] | 49.12% |
| 8B | humaneval | 37.04 | 67.86 | 1.832 [1.784, 1.889] | 34.11% |
| 8B | mbpp | 36.94 | 67.53 | 1.828 [1.794, 1.868] | 33.95% |
| 8B | mtbench | 36.99 | 42.91 | 1.160 [1.088, 1.247] | 16.83% |

Every evaluated workload accelerates over paired AR, with all bootstrap intervals above one. MT-Bench has shorter accepted progress and smaller gains. This is consistent with the measured acceptance counts, but does not establish a causal explanation for the domain difference. EAGLE-3 14B source reuse remains faster on HumanEval and MBPP. Source removal trades accepted progress against cycle cost. GSM8K scores are 119/128 versus AR 120/128 at 8B and 122/128 for both at 14B. New code and conversation outputs have not yet received their corresponding quality evaluation. Historical code scores cannot be substituted for these outputs.

## Distinct data at fixed computation

Train separate DFlash 4B-to-8B maps on nested sets of 512, 1,024 or 2,048 records. Keep 1,024 updates, four records per update, objective, seed and optimizer settings fixed. Every map receives 4,096 presentations. The 4,096-distinct-record cell will reuse the continuous run at update 1,024. Each map is evaluated on the same 128 MATH-500 requests with paired AR, native DFlash, source reuse and RelaySpec.

| Distinct records | Relay tokens/s | AR speedup [95% CI] | Native throughput retained | Acceptance | Progress/cycle |
| ---: | ---: | --- | ---: | ---: | ---: |
| 512 | 181.91 | 4.836 [4.623, 5.054] | 88.93% | 38.47% | 6.770 |
| 1024 | 184.55 | 4.918 [4.695, 5.146] | 90.29% | 39.19% | 6.878 |
| 2048 | 183.81 | 4.883 [4.658, 5.115] | 89.82% | 38.95% | 6.842 |

The completed cells show similar performance rather than monotonic gains from more distinct data. They support useful performance from a small repeated fitting set on this development-exposed evaluation subset. They do not establish a best data size or generalization beyond this setting. Each scores 105/128 versus 104/128 for its paired AR arm. Single-seed, paired-request intervals do not measure fitting-seed uncertainty.

## Continuous fitting

Fit one map for 2,048 updates without resetting AdamW. Save updates 128, 256, 512, 1,024 and 2,048. These correspond to 512, 1,024, 2,048, 4,096 and 8,192 presentations. The final point repeats the same 4,096 distinct records. Checkpoint optimizer counters passed. The actual warm fitting loop took 193.46 seconds. The long job time is dominated by four-method generation evaluation at each checkpoint.

The completed 128-update checkpoint reaches 138.07 tokens/s, 3.653x AR [3.482, 3.827], 67.44% native throughput retained, 27.52% draft acceptance and 5.128 tokens of progress per cycle. Its math score is 105/128 versus AR 104/128. Later checkpoint results are not yet complete. Comparing this early 512-record checkpoint with the repeated-512-record fit suggests additional optimization helps, to be checked against the complete trajectory.

## Existing experiments reused in the manuscript

| Experiment | Procedure and purpose | Existing finding |
| --- | --- | --- |
| Main MATH-500 | Four drafter/target combinations, each with 500 paired requests. Establish acceleration over AR. | DFlash 8B/14B: 4.99x/5.11x. EAGLE-3 8B/14B: 2.35x/2.59x. Already complete, no rerun needed. |
| Native drafter reference | Compare reused 4B head with the target’s own released head in paired runs. | 90.3% DFlash-8B, 90.2% EAGLE-3-8B, 89.1% EAGLE-3-14B throughput retained. |
| DFlash workload breadth | AR/source/relay across math, code and conversation at both target sizes. | All eight AR comparisons accelerate, spanning 1.85x–3.69x. |
| Source removal | Compare relay with the same frozen head conditioned by its source transformer. | Faster main math runs despite reduced accepted progress. Not universally faster across all workload cells. |
| Isolated memory | Measure source reuse and relay in separate EAGLE-3 processes. | Peak allocated memory falls by 7.80 GiB at 8B and 7.63 GiB at 14B on RTX 6000 Ada. |
| Historical fitting sweep | Vary records and training duration together. | Throughput increases from 136.78 to 187.85 tokens/s. The new controlled runs separate distinct data from optimizer work. |
| Block length | Compare 8, 16 and 32 draft positions, paired with AR and native DFlash on 64 questions. | Block 16 gives the highest measured relay throughput, 210.15 tokens/s. Longer blocks need not be faster. |
| EAGLE-3 normalization | Fixed data/update budget, raw versus normalized input features. | Raw features reduce relative context error from 0.413 to 0.266 and improve throughput relative to source reuse from 1.035x to 1.237x. |
| Objective and map alternatives | Compare relative-error fitting, squared-error plus cosine, ridge regression and nonlinear maps. | Selected relative-error objective improves throughput about 1% over squared-error plus cosine in the matched comparison. Ridge/nonlinear studies change multiple recipe elements and are not isolated causal tests. |
| Numerical agreement and task quality | Score timed outputs and trace identical prefixes around token differences. | Main math quality differs from AR by -0.2 to +0.8 percentage points. Selected traces show finite-precision call-shape differences. This does not establish bitwise output identity. |
| Data overlap | Compare fitting and evaluation problem text. | Zero normalized exact matches. Related templates remain, including 47 MATH-500 questions above 0.80 token-set overlap. |

## Next actions

1. Finish continuous checkpoint evaluations and collect every completed cell. Reuse update 1,024 for the missing 4,096-record fixed-compute point.
2. Plot data size and continuous training separately, with paired AR speed, native retention and acceptance. Do not label 8,192 presentations as distinct examples.
3. Evaluate the saved new code outputs with the existing isolated evaluator. Conversation quality requires its own evaluation rather than a speed proxy.
4. Run additional fitting seeds after the trajectory determines a fixed recipe. Use a short pilot first and keep the total allocation at four GPUs.
5. Follow with a matched-budget drafter-adaptation control to test the choice to train only the interface. The capability table alone is not empirical superiority evidence.
6. Promote reviewed results into the paper asset registry and regenerate the PDF. Preserve all planned workload cells and uncertainty.

## Validation

The completed cells above passed their completion gates, record-count checks and raw-file SHA-256 checks. The pinned scorer analysis is present for saved math outputs. The ongoing job is a batch allocation and remains intentionally active. These result snapshots do not alter the manuscript’s existing evidence registry.
