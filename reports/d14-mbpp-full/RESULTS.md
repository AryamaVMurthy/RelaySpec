# Qwen3-14B + DFlash-4B: official full MBPP

## Protocol

- Dataset: all 378 EvalPlus MBPP tasks at revision
  `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`.
- Target: `Qwen/Qwen3-14B` at revision
  `40c069824f4251a91eefaf281ebe4c544efd3e18`.
- Proposer: `z-lab/Qwen3-4B-DFlash-b16` at revision
  `b74e3a329c4d963783143b1e970d95b002be72bd`.
- Non-thinking, greedy decoding, 2,048-token generation cap, draft block 16.
- Hardware: exactly four NVIDIA RTX 6000 Ada GPUs on Turing `node01`.
- The first 200 and disjoint final 178 tasks were generated in jobs 25491 and
  25506, then exhaustively validated and merged before official EvalPlus
  scoring in job 25510.

## Main result

| Method | End-to-end tok/s | vs native AR | vs source DFlash | Mean accepted | EvalPlus base | EvalPlus Plus |
|---|---:|---:|---:|---:|---:|---:|
| Native AR | 26.716 | 1.000x | 0.310x | 1.000 | 88.10% | 74.07% |
| Source-trunk DFlash | 86.292 | 3.230x | 1.000x | 5.999 | 88.36% | 75.13% |
| RelaySpec | **89.880** | **3.364x** | **1.042x** | 4.231 | **88.36%** | **75.13%** |

RelaySpec's paired end-to-end speedup over source-trunk DFlash is **1.0416x**
with a 95% bootstrap CI of **[1.0289, 1.0536]**. It reproduces the source
method's generated sequence on **378/378 tasks**, so its official functional
accuracy is identical to the source baseline on this run. The output cap was
hit on 1/378 tasks for every method.

The native-AR row is an independent greedy decode and therefore need not emit
the same code as speculative decoding, even when both are valid. Accuracy is
reported with the official EvalPlus base and Plus test suites, not sequence
matching to native AR.

## Profiling interpretation

The source baseline spends 29.91% of request time recomputing the Qwen3-4B
source trunk during verification. RelaySpec removes that region, replacing it
with 0.63% relay work. The saved work is partly reinvested in more verifier
cycles because acceptance falls from 5.999 to 4.231 tokens, leaving a modest
but statistically positive 4.16% end-to-end improvement.

## Reproducibility artifacts

- `config.yaml`: frozen protocol and model revisions.
- `eval-manifest.json` and `eval-manifest.sha256`: exact 378 prompts.
- `benchmark-rank*.jsonl`: merged per-request raw measurements.
- `merge-provenance.json`: shard hashes and completeness validation.
- `benchmark-paper-summary.json`: paired performance aggregates and CIs.
- `evalplus-summary.json`: official base/Plus pass@1 counts.
- `evalplus/*-full.eval_results.json`: complete official scorer results.
- `first-allocation.jsonl`, `remainder-allocation.jsonl`, and GPU CSVs:
  allocation and utilization provenance.

Job 25510's Slurm state is `FAILED` only because the first summary wrapper
counted Plus-only passes without also requiring base-test success. The official
EvalPlus scorer had completed successfully. The wrapper was corrected with a
regression test and `evalplus-summary.json` was regenerated from the unchanged
official result files; no model output or score was rerun or altered.
