# Faster Llama transfers with exact matched-AR outputs

Recommended established mapper settings: FP32 target and mapper, BF16 frozen
drafter and source embedding/head. Use block size 10 for Llama-3.1-8B DFlash
to Llama-3.2-3B, and block size 16 for Qwen3-4B DFlash to Llama-3.1-8B.
The verifier and plain-AR code remain unchanged.

## Improvement on identical eight prompts, cap 1024

| Transfer | Previous full-FP32 relay TPS | Mixed relay TPS | Improvement | Matched FP32 AR TPS |
|---|---:|---:|---:|---:|
| Llama to Llama | 91.43 | 106.27 | 16.2% | 46.25 |
| Qwen to Llama | 61.66 | 64.28 | 4.2% | 22.01 |

These comparisons have identical prompt sets, caps, and output hashes.
Full-FP32 reference is job 28885; new mixed run is 28922. Request-time
improvements, including prefill, are 16.1% and 4.2%. Paired request bootstrap
intervals are saved in `mixed-improvement.json`; they exclude between-run
timing variability and fitting-seed uncertainty.

The alternative Llama block16 reached 102.83 TPS; cross-family block8 reached
54.69 TPS. Both matched AR 8/8, but were slower than the selected sizes.

## Fresh eight-prompt confirmation, cap 512

| Transfer / block size | Relay TPS | Matched AR TPS | Exact output matches |
|---|---:|---:|---:|
| Llama / 10 | 117.36 | 46.88 | 8/8 |
| Llama / 6 | 110.47 | 47.04 | 8/8 |
| Cross-family / 16 | 45.95 | 22.29 | 8/8 |
| Cross-family / 8 | 41.69 | 22.28 | 8/8 |

The selected settings each match FP32 AR on all 16 distinct prompts in this
pass. The two sets have different caps and are reported separately. This is
empirical output equivalence on tested requests, not proof for every possible
input. It does not preserve the earlier BF16 AR output stream. Exact AR
agreement is distinct from mathematical-answer accuracy.

## Training counts

| Mapper | Distinct training records | Epochs/passes | Optimizer updates |
|---|---:|---:|---:|
| Established mapper, both transfers | 4096 | 1 | 1024 |
| ZIP pilot, both transfers | 512 | 3 | 54 |
| Longer ZIP fit, both transfers | 512 | 20 | 360 |

The old trainer used one example per GPU per update on four GPUs. ZIP used
2048 sampled feature positions per batch, so epoch and update counts are not
directly comparable across recipes. Both old checkpoint files independently
confirm 1024 updates; the manifest contains 4096 distinct normalized problems.
No mapper was retrained in this speed pass. Improvements are from inference
precision and block-size choices. The ZIP training budgets differ, so this
does not isolate the loss-function effect.

Training versus evaluation exact normalized problem overlap is zero for
both eight-prompt sets; no semantic near-duplicate audit is implied.

Jobs 28922 and 28923 each ran four independent single-GPU lanes. Configs,
raw output hashes/text, timings, and numerical dtype metadata are saved beside
this report. A final runtime-matched ZIP screen is separately limited to four
prompts per family (job 28925).

Final ZIP screen: both pairs matched FP32 AR on 4/4 prompts. Matched-runtime, matched-prompt results:
- llama: old mapper 110.42 TPS; ZIP20 94.02 TPS.
- cross: old mapper 52.96 TPS; ZIP20 49.09 TPS.

All three jobs completed with exit code 0 (6m14s, 4m15s, 1m29s). No GPU jobs remain.
