# Initial results

Job 28841 completed successfully in 4m26s using four L40S GPUs.
Eight MATH development prompts per setting, maximum 512 output tokens.
Numbers below are decoding TPS, total output tokens divided by total decode time.

| Transfer | AR in ZIP run | ZIP 3 epochs | Old mapper | ZIP exact token match to AR |
|---|---:|---:|---:|---:|
| Llama-3.1-8B drafter to Llama-3.2-3B | 61.32 | 98.52 | 164.92 | 4/8 |
| Qwen3-4B drafter to Llama-3.1-8B | 41.54 | 41.78 | 96.33 | 0/8 |

AR was remeasured in the old-mapper runs: 62.58 and 41.51 TPS, respectively.
The old Llama mapper also matched AR on 4/8 requests; the old cross-family
mapper matched on 0/8. Different output sequences and lengths mean these
TPS ratios are not established lossless speedups. The new Llama mapper
matched all eight source-reuse outputs, but source reuse itself disagreed
with AR on four. No confirmed root cause is claimed.

Both new fits used 512 existing calibration records and 54 updates over
three epochs. Llama sampled 36,650 positions; cross-family sampled 36,605.
Final training objectives: 1.23096 and 1.55160. Training including final
export/check bookkeeping took 4.06s and 4.27s. BF16 folding relative MSE
was 1.85e-5 and 1.90e-5, below the 1e-3 export gate.

The initial pilots do not improve the previous mappers. They also do not
isolate objective choice: the old fits had different optimization/data
budgets, and these pilots use plain solution text rather than the ZIP
generated-rollout pipeline. This is not evidence that the loss cannot
work with adequate fitting.

A 20-epoch, freshly initialized control reusing the exact cached features
was submitted as job 28857, with a new 20-epoch cosine schedule. It is an
optimization-budget ablation, not the original three-epoch ZIP protocol.
Its measurements are pending; no results are implied here.

Raw per-request records and summaries: `run-28841/{llama,cross}-{zip,old}`.
Configuration and source: this directory and `scripts/train_zip_family_pilot.py`.
