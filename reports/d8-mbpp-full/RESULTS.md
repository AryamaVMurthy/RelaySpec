# Qwen3-8B + DFlash-4B: official full MBPP

## Protocol

- Dataset: all 378 EvalPlus MBPP tasks at revision
  `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`.
- Target: `Qwen/Qwen3-8B` at revision
  `b968826d9c46dd6066d109eabc6255188de91218`.
- Cross-scale proposer: `z-lab/Qwen3-4B-DFlash-b16` at revision
  `b74e3a329c4d963783143b1e970d95b002be72bd`.
- Target-specific reference: `z-lab/Qwen3-8B-DFlash-b16` at revision
  `9b41424b7109f9c5413454f481b09a82b85333f4`.
- Non-thinking, greedy decoding, 2,048-token generation cap, draft block 16.
- Hardware: exactly four NVIDIA RTX 6000 Ada GPUs on Turing `node01`.
- The first 200 and disjoint final 178 tasks were generated in jobs 25493 and
  25507, then exhaustively validated, merged, and scored in CPU job 25513.

## Main result

| Method | End-to-end tok/s | vs native AR | vs source DFlash | Mean accepted | EvalPlus base | EvalPlus Plus |
|---|---:|---:|---:|---:|---:|---:|
| Native AR | 44.529 | 1.000x | 0.385x | 1.000 | 84.13% | 73.02% |
| Source-trunk DFlash | 115.686 | 2.598x | 1.000x | 6.217 | 84.13% | 73.02% |
| Target-specific DFlash-8B | **201.060** | **4.515x** | 1.738x | **6.662** | 84.13% | 73.02% |
| RelaySpec | 146.195 | 3.283x | **1.264x** | 4.582 | 84.13% | 73.02% |

RelaySpec's paired end-to-end speedup over source-trunk DFlash is **1.2637x**
with a 95% bootstrap CI of **[1.2497, 1.2773]**. Source reuse, target-specific
DFlash, and RelaySpec produce the same token sequence on **378/378 tasks**, so
all have exactly the same official functional accuracy. One speculative output
per method reached the 2,048-token cap; native AR reached none.

RelaySpec reaches 72.7% of the throughput of target-specific DFlash-8B while
using the reusable DFlash-4B proposer rather than a separately trained
target-specific proposer. The target-specific baseline remains the throughput
upper reference when its checkpoint is available.

## Profiling interpretation

The source baseline spends 38.28% of request time recomputing the Qwen3-4B
source trunk during verification. RelaySpec removes that region, replacing it
with 0.78% relay work. Its acceptance declines from 6.217 to 4.582 tokens, so
verifier work rises, but the net request-level gain remains a statistically
clear 26.37%.

## Reproducibility artifacts

- `config.yaml`: frozen protocol and model revisions.
- `eval-manifest.json` and `eval-manifest.sha256`: exact 378 prompts.
- `benchmark-rank*.jsonl`: merged per-request raw measurements.
- `merge-provenance.json`: input hashes and exhaustive cell validation.
- `benchmark-paper-summary.json`: paired performance aggregates and CIs.
- `evalplus-summary.json`: official base/Plus pass@1 counts.
- `evalplus/*-full.eval_results.json`: complete official scorer results.
- Allocation JSONL, GPU CSV, and source-hash files: hardware and code
  provenance for both generation shards.

The original dependent scorer job 25509 failed before reading model outputs
because its queued wrapper lacked exported input paths. CPU-only job 25513 was
resubmitted with explicit frozen paths and completed successfully. No model
generation was repeated or modified.
