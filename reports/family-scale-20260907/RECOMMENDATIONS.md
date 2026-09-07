# Recommended settings after expanded calibration

Cross-family transfer improved on the reserved evaluation. The larger Llama-to-Llama fit did not improve held-out speed meaningfully.

| Reused drafter → target | Recommendation | Block | Reserved decode TPS | AR TPS |
|---|---|---:|---:|---:|
| Llama-3.1-8B DFlash → Llama-3.2-3B | Keep existing 4,096-record, one-pass mapper | 10 | 113.63 | 46.54 |
| Qwen3-4B DFlash → Llama-3.1-8B | New dense mapper: 8,192 records, epoch-3 checkpoint | 16 | 67.84 | 22.04 |

The new cross-family mapper improves decode throughput from 58.85 to 67.84 tokens/s (15.3%), reaching 3.08× matched AR throughput. Including prefill, throughput improves from 58.03 to 66.71 tokens/s (15.0%). The paired request-bootstrap 95% interval for this gain is 10.8–19.9%; it excludes run-to-run and fitting-seed uncertainty.

The selected 16,384-record Llama mapper reaches 113.66 tokens/s versus 113.63 for the existing mapper. Its request-speedup interval is 0.977–1.025. The apparent development gain did not carry over; retain the cheaper existing calibration.

Both comparisons use 16 reserved MATH requests, up to 1,024 generated tokens, with identical output hashes between old mapper, selected mapper and matched FP32 AR. Actual generated-token totals are 6,724 for Llama and 8,477 for cross-family. Exact agreement is a finite-request check, not an answer-quality score or a universal floating-point guarantee.

## What was tested

- 4,096 / 8,192 / 16,384 nested Numina records; 256 common fitting-validation records.
- Dense relative interface loss and the archived equal-weight ZIP layer-plus-context loss on identical cached features.
- Twelve initial fits, saving epochs 1/3/6/12; four adaptive fits at lower LR or 24 epochs.
- 74 short checkpoint/control arms; 24 coarse block arms; 16 longer development arms; four final old/selected arms. Arms reuse requests and are not independent datasets.
- Coarse blocks 6–32, followed by 9/10/11 for Llama and 15/16/17 for cross-family.
- Exploratory FP16 frozen-drafter pilots retained exact outputs but gave no consistent timing gain; these overlapped fitting and were not used for promotion. BF16 remains the selected drafter/interface precision.

The cross-family winner is epoch 3 (384 updates) **from a 24-epoch cosine schedule**, not a separately scheduled three-epoch run. Its fresh initialization is seed 42, peak LR 0.001, AdamW with zero weight decay, 5% warmup, clipping 1, and 64 records per update. Each record contributes at most 32 cached shared-token-end positions; an epoch revisits those positions. Only the mapper trains. The winning checkpoint has 786,414 sampled-position exposures.

The data consist of supplied problem/solution text, truncated at 1,024 tokens per tokenizer; this campaign is not a generated-rollout reproduction. Source manifests are pinned, hash-checked and lexically filtered against the MATH reference. This does not establish semantic/template independence. The new 4k control uses the new Numina recipe; historical 4k fitting used a different recipe.

## Reproduction

Persistent recommended checkpoints are under:

`/home/aryama.murthy/rs-zip-family-20260907/release/family-scale-20260907/`

Ready-to-run configs are in `configs/protocol_next/family-speed-20260907/`. `run.sbatch` uses four GPUs, with two independent repetitions per model pair. It benchmarks AR alongside the recommended mapper and checks exact output agreement. It has been prepared but not launched as an unnecessary repeat of the completed confirmation.

The target and mapper run in FP32 with TF32 disabled; frozen drafter and source embedding/head run in BF16. Deployment presets preserve the measured runtime. The new dense fitter used input RMS epsilon 1e-6; the existing runtime uses the target's input epsilon 1e-5 and frozen output epsilon 1e-6. A fixed-feature audit over 256 validation records per pair measured mean output-context relative MSE about 5.1–5.5e-7 between those input-epsilon choices. This is documented in `artifacts/normalization-audit.json`; no runtime epsilon or weights were changed after selection.

Checkpoint SHA-256:

- Llama recommendation: `0c9e91c35d037839bc1693f0ce78a087c0b2304831a040309824ac8bd0f465e7`
- Cross-family recommendation: `7ad23f549833b34cbe25e588ccd8cb6980306e33defce755cddb56206030da50`

## Paper implications

The positive result is improved cross-family reuse with more calibration data, supported by a matched reserved comparison. Within-family saturation and the failure of lower feature loss to guarantee higher decoding throughput are equally useful findings. ZIP did not outperform dense in the development sweep. The two-request scaling plots are exploratory and must not be presented as independent confirmatory benchmarks.

Raw outcomes, full fitting histories and uncertainty calculations are in `results.json`. Figures include validation curves, development throughput curves and the reserved request-speedup comparison.

Every successful GPU job finished in under ten minutes. The four-lane 16k/12-epoch fit took 5m32s; the adaptive fitting job took 9m44s; final confirmation took 9m24s. Total allocated GPU-job wall time, including corrected pilots, was 66m57s on four GPUs. This is search cost, not the cost of fitting the selected checkpoint.
