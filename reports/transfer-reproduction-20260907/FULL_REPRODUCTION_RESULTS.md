# Full transfer reproduction and TPS comparison

The full 16,384-record, three-epoch transfer mapper achieved **167.83–167.97 tokens/s**, compared with **163.57–163.67tokens/s** for native DFlash and **26.05tokens/s** for normal autoregressive decoding. Its approximately 2.6% advantage over native held in a second run with GPU assignments reversed. These results reproduce the archive's inference behavior and approximate performance on newly generated training data; they do not reproduce the historical training data or mapper tensor hashes bit for bit.

## Matched throughput results

128 original evaluation prompts; maximum 2,048 output tokens/request;124,899 actual output tokens/method/run;104 normal stops and 24 length stops. Single measured request per GPU, BF16, L40S, greedy/no-thinking, prefix cache disabled, batch-invariant inference. TPS is total actual output tokens divided by summed per-request generation wall time, including prefill and verification; model setup and four warmups are excluded. Requests affected by new compilation are rerun once and their cold time is retained in raw logs. Four GPUs parallelize independent requests; this is not four-GPU batched serving throughput.

| Method | Run0 TPS | Run1 TPS, GPU assignment reversed | Speedup vs AR |
|---|---:|---:|---:|
| Normal autoregressive8B |26.0503|Same baseline, not rerun|1x|
| Native8B DFlash |163.5698|163.6735|6.279–6.283x|
|4B-drafter transfer to8B |167.8297|167.9705|6.443–6.448x|

Mapped/native ratio: **1.02604** in run 0, paired-request 95% interval **[1.01801,1.03397]**; **1.02625** in run 1, interval **[1.01824,1.03417]**. Intervals use 10,000 paired request bootstrap resamples, seed 42. They describe request variation within each run; they do not measure fitting-seed or population-wide hardware variance. The observed two-run variation is reported explicitly, not hidden inside a pooled confidence interval.

Pooling tokens and time across both speculative runs gives native **163.6216TPS** and mapped **167.9001TPS** (ratio **1.02615**). Both runs are retained; no best-run selection.

## Correctness and fidelity

- All 128prompts, output-token arrays and finish reasons match between AR, native and mapped in each run. All output arrays also match the archived reference. The repeated speculative outputs match the same AR baseline exactly.
- Target/drafter embedding and output-head attachment probes passed for every worker. The frozen source fusion and RMSNorm tensor hashes match the archive.
- The full run uses byte-identical archive mapper/loss, optimizer/training, export and benchmark source. Only capture uses eager execution and batch invariance: compiled capture failed its 1e-6 sampled/dense check; the corrected capture gives zero error on all 16 representative comparisons. No tolerance was relaxed. See source/source-audit.json.
- All 128training rollout shards differ from historical SHA256 values. Pinned dataset, splits, evaluation prompts, model revisions and software versions matched; GPU driver 570.211.01 differs from archive 610.57.04. Queue-sensitive floating-point behavior is a plausible contributor, not a demonstrated complete explanation. The new realization was explicitly accepted and preserved. Mapper hashes likewise differ, so exact historical weight reproduction is not claimed.
- Output equality on these 128prompts demonstrates behavior on this benchmark, not universal equality or quality across arbitrary tasks. Training uses one fitting seed 42; no seed sweep or quality-based checkpoint selection was performed.

## Training

16,384 distinct freshly generated NuminaMath records, up to 4,096 output tokens each. Both frozen targets see identical complete sequences. Segment-stratified 25% token sampling yields 5,398,695 positions/epoch. Five bias-free4096-to2560maps at layers [1,9,17,25,33],52,428,800 trainable parameters, Xavier initialization, seed 42. Equal-weight per-layer relativeMSE plus frozen fused-context relativeMSE; each example has equal total weight. AdamW lr 1e-3, zero weight decay, 5% warmup then cosine decay, batch 2,048 positions, gradient clipping 1; all non-mapper weights frozen. Three fixed epochs, final checkpoint; no validation-based selection.

| Epoch | Loss | Updates cumulative | Epoch time |
|---|---:|---:|---:|
|1|0.737848|2,637|281.13s|
|2|0.329735|5,274|293.07s|
|3|0.239859|7,911|290.66s|

Epoch compute: 864.86 seconds (~14m25s). Capture plus complete paired-feature validation:51m58s; training/audit/export/eight-prompt pipeline:15m50s; full comparison31m47s; speculative repeat7m43s. Generation was already completed in96m25s; its job reported failure at the historical-hash comparison after all16,384records were saved. These are separate stage timings, not mapper-training time.

Folded BF16 export relative context MSE **2.1481e-5**, below 1e-3 tolerance. Final checkpoint locally preserved at `/home/aryamavmurthy/work/RelaySpec-scaling/artifacts/transfer-reproduction-20260907/full 16384-final.pt`; local/remote SHA256 **6041cacaabad12f590a5108d1b5df537315734f3fee07a3af243f2558d4cc2cf**.

## Earlier small-data pilot

With 512 records and three epochs (243 updates), mapped159.76TPS vs175.55native and26.24AR on 8 prompts capped 512 tokens. Full-data fitting raised mapped throughput on those same 8 prompts to179.35TPS. This changes both distinct records and total optimization work, so it is not an isolated data-scaling or loss-function ablation. The 128-prompt results above provide the requested full-data comparison.

## Evidence

- `full 16384-results/final-comparison-r0.json` and `final-comparison-r1.json`: audited statistics, intervals, source hashes and GPU assignments.
- `full 16384-results/final/`: per-request raw outputs and worker metadata for all methods and repeats.
- `full 16384-results/fit-summary.json`, `feature-manifest.json`, `mapper-audit.json`, `export-check.json`, `checkpoint-preservation.json`.
- `full 16384-results/source/`: exact executed package source and audit against the ZIP.
- `jobs.json`: Slurm jobs28719,28720,28721,28735and associated launch scripts. All four completed with exit 0.
