# Qwen3-14B MATH-500 result

This is the first complete RelaySpec benchmark: 500 fixed MATH-500 prompts,
Qwen3 non-thinking mode, greedy decoding, a 2,048-token cap, block size 16, and
exactly four RTX 6000 Ada GPUs.

## Result

| Method | Accuracy | Output tok/s | Speedup vs source reuse | 95% paired CI | Mean accepted | Cap rate |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-14B native AR | 79.2% | 26.7 | 0.244x | [0.238, 0.249] | 1.00 | 8.8% |
| 4B source trunk + DFlash + 14B verifier | 79.6% | 109.7 | 1.000x | [1.000, 1.000] | 7.61 | 9.4% |
| RelaySpec feature relay + DFlash + 14B verifier | 79.6% | 143.7 | **1.309x** | **[1.302, 1.316]** | 6.80 | 9.4% |

Including all per-request overhead, RelaySpec achieves 142.4 output tok/s and
**1.311x** end-to-end speedup over source reuse, with a 95% paired confidence
interval of **[1.304, 1.318]**.

RelaySpec and source reuse produce the same sequence on all 500 prompts and
therefore have exactly the same official accuracy. Their accuracy difference is
0.0 percentage points (95% paired CI [0.0, 0.0]). Relative to native AR, both
are +0.4 percentage points, but the confidence interval [-1.4, +2.4] percentage
points includes zero. Equality to native AR is only 23.2%; this is reported as a
floating-point/kernel-shape diagnostic, not as a task-quality claim.

For this target, the same relay form is a single bias-free projection from five
5,120-wide target taps to the 2,560-wide proposer interface, or 65,536,000
trainable parameters. The target and proposer remain frozen.

## Runtime decomposition

RelaySpec request-time shares are:

- full-target verification: 88.47%;
- DFlash proposal: 9.20%;
- relay plus relay prefill: 0.64%;
- full-target prefill: 0.88%;
- unattributed runtime: 0.82%.

Source reuse spends 30.11% of request time in its source trunk (prefill plus
verification). RelaySpec removes that trunk at the cost of 0.64% relay work,
but accepts 6.80 rather than 7.61 tokens per target verification. The measured
1.31x speedup is therefore an end-to-end result after paying the acceptance
penalty, not a component-only projection. The remaining bottleneck is target
verification, so future optimization must improve accepted tokens per verifier
call rather than further shrinking the already-small relay.
The measured 30.11% removable fraction implies a 1.431x ideal Amdahl ceiling;
the observed 1.311x end-to-end result realizes 91.6% of that ceiling.

Length stratification further localizes the deployment regime. Decode speedup
over source reuse is 1.254x at up to 256 output tokens, 1.271x at 257--512,
1.319x at 513--1,024, and 1.321x at 1,025--2,048. RelaySpec therefore becomes
more valuable, not less, on the longer generation workloads targeted by this
paper.

## Provenance

- Generation Slurm job: `25376`; official-scoring job: `25426`.
- The generation job wrote all 1,500 paired records and its aggregate summary,
  then Slurm marked it failed because a fast rank waited more than NCCL's former
  10-minute barrier timeout. All four rank files contain exactly 375 records.
  The post-generation timeout cannot affect the complete paired outputs or
  scores. The harness now uses a six-hour process-group timeout.
- Official scorer: pinned Qwen math evaluator with 10,000 paired bootstrap
  replicates.
- Immutable artifacts: `benchmark-summary.json`,
  `benchmark-paper-summary.json`, and `config.yaml` in this directory.
- Raw paired evidence: four rank files, 375 rows each, 1,500 total records,
  stored under `raw/`.
- The checkpoint in this result uses refined feature reconstruction. Its legacy
  raw label is `relay_p`; the paper-facing name is `relay_f_refined` until the
  proposer-aligned objective is trained and evaluated.
