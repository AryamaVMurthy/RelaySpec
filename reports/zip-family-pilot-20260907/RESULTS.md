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

A 20-epoch, freshly initialized control reused the exact cached features
in job 28857, with a new 20-epoch cosine schedule and 360 updates. It is an
optimization-budget ablation, not the original three-epoch ZIP protocol.

| Transfer | AR | ZIP 20 epochs | Old mapper | ZIP exact token match to AR |
|---|---:|---:|---:|---:|
| Llama-3.1-8B drafter to Llama-3.2-3B | 61.36 | 139.02 | 164.92 | 4/8 |
| Qwen3-4B drafter to Llama-3.1-8B | 41.54 | 94.22 | 96.33 | 0/8 |

Twenty-epoch training took 22.69s for Llama and 18.50s for cross-family;
final objectives were 0.57723 and 0.74239. More optimization substantially
improved throughput, but neither new map beats the old mapper in these
pilots. Output-agreement concerns remain unchanged. The data support a
fitting-budget sensitivity finding, not a new positive generalization claim.

These eight-prompt measurements are exploratory, single-fit observations;
small timing differences are not established improvements. Larger generated
rollouts and a decoder correctness investigation remain separate future work.

Raw per-request records and summaries: `run-28841/{llama,cross}-{zip,old}`.
Twenty-epoch records: `run-28857/{llama,cross}-epoch20`.
Configuration and source: this directory and `scripts/train_zip_family_pilot.py`.
