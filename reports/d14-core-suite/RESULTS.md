# Qwen3-14B cross-task suite

Generation job `25437` and scoring jobs `25438`/`25439` evaluate 128 GSM8K,
164 HumanEval, and 200 MBPP prompts in Qwen3 non-thinking greedy mode with a
2,048-token cap and exactly four GPUs. RelaySpec and source reuse produce
identical sequences on every one of the 492 prompts.

| Benchmark | Metric, source / relay | Source reuse tok/s | Relay tok/s | Relay/source end-to-end speedup | Mean accepted, source / relay |
|---|---:|---:|---:|---:|---:|
| GSM8K-128 | 94.53% / 94.53% | 86.4 | 103.2 | **1.194x [1.176, 1.214]** | 6.01 / 4.88 |
| HumanEval | 89.02% / 89.02% base; 83.54% / 83.54% Plus | 85.6 | 83.2 | 0.972x [0.957, 0.987] | 5.89 / 3.90 |
| Legacy MBPP-200 | invalid quality protocol | 75.6 | 73.5 | 0.972x [0.953, 0.990] | 5.25 / 3.40 |

Native AR scores 94.53% on GSM8K and 90.85% base / 85.98% Plus on HumanEval.
The relay-specific quality delta is exactly zero because all source/relay
outputs match. HumanEval uses pinned EvalPlus revision
`26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`.

The legacy MBPP manifest omitted the published assertions needed to identify
required function names and only partially overlaps current EvalPlus task IDs.
Its 6% resource-scorer value is therefore invalid for paper quality and is not
reported as a result. A replacement 200-task manifest sampled directly from
the pinned EvalPlus registry is queued for a complete D14 rerun with official
base and Plus scoring.

The runtime result establishes the method's current boundary. The MATH-trained
relay remains beneficial on GSM8K reasoning, but loses acceptance on code and
is about 2.8% slower than source reuse there. This is not a verifier-accuracy
failure; it is proposer conditioning under domain shift. The paper's positive
claim is therefore math/reasoning inference unless mixed-domain relay training
passes a held-out code throughput gate.

Raw evidence contains four 369-row rank files under `raw/`; official scorer
artifacts are under `evalplus/`.
