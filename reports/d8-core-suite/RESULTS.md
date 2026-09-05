# Qwen3-8B cross-task suite

Generation job `25440` completed in 33m49s on exactly four RTX 6000 Ada
GPUs. It produced four 492-row rank files: 1,968 records covering 128 GSM8K,
164 HumanEval, and 200 legacy-MBPP prompts under four paired methods. Every
RelaySpec completion is token-identical to both source reuse and the official
target-specific DFlash baseline.

| Benchmark | Quality, source / relay | Source tok/s | Relay tok/s | Relay/source end-to-end speedup | Accepted, source / relay |
|---|---:|---:|---:|---:|---:|
| GSM8K-128 | 93.75% / 93.75% | 114.5 | 163.8 | **1.431x [1.412, 1.450]** | 6.13 / 5.13 |
| HumanEval | 85.98% / 85.98% base; 80.49% / 80.49% Plus | 116.6 | 139.3 | **1.195x [1.176, 1.213]** | 6.17 / 4.31 |

On GSM8K, official target-specific DFlash reaches 195.6 tok/s and RelaySpec
reaches 83.7% of that throughput while avoiding target-specific proposer
training. On HumanEval those values are 202.4 tok/s and 68.8%. Native AR is
44.8 and 44.7 tok/s respectively. The source/relay quality delta is exactly
zero because all paired sequences match.

HumanEval is scored by pinned EvalPlus revision
`26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`, with both base and Plus tests.
The native-AR row scores 88.41% base and 83.54% Plus; its numerical differences
from block verification are not caused by the relay.

The original MBPP timing rows came from a Google sanitized-test manifest that
omitted required function-name assertions and whose task IDs only partially
overlap current EvalPlus. They are retained as immutable diagnostic evidence,
but are not a paper-quality result. The replacement manifest
`configs/eval_manifest_full_v3.json` samples 200 tasks directly from the pinned
EvalPlus registry, includes every base assertion in the prompt, and is queued
for a clean four-GPU rerun and official base/Plus scoring.
