# Near-peak exact AR/speculative agreement

User requested on 2026-09-12: retain execution close to peak throughput while
obtaining exact greedy token sequences between AR and speculative decoding.
The 128-request optimized rerun (32600, collector 32604) was cancelled to resolve
this first. Existing results and partial files are retained.

## Contract

Use frozen existing models and checkpoints, BF16, greedy sampling, identical
prompt tokens, natural EOS, one active request per GPU. Four L40S maximum on
node07. No training, quantization, altered logits, forced reference tokens, or
accepting a close text match as an exact match. Comparison means the entire
output-token-ID sequence and finish reason match. Matching a slower common
runtime is distinct from matching stock-O3's particular floating-point outputs.

Diagnostics use the first eight main-cohort Qwen3-8B requests and a 512-token
cap plus one warmup request. They are runtime diagnostics, not final paper
measurements. Final confirmation must use 128 requests and a 2,048-token cap
on each target, and include all requested speculative checkpoints.

## First controlled wave: 32605 (completed)

All profiles used O3, V2 runner, asynchronous scheduling and CUDA graphs.

| Profile | AR TPS | Native DFlash TPS | Exact sequences / 8 |
|---|---:|---:|---:|
| Stock O3 | 46.22 | 210.22 | 2 |
| Batch-invariant O3 | 27.44 | 139.63 | 3 |
| Strict BLAS precision / workspace | 46.06 | 213.69 | 1 |
| Batch-invariant, smaller matrix tiles | 41.89 | 205.06 | 2 |

The smaller-tile kernel passes bitwise prefix-row tests at M=1,8,16,32 versus
M=129 for four relevant matrix shapes, including the output head. This checks
one kernel, not the complete model. It reaches 90.6% of stock AR throughput
and 97.5% of stock native throughput on this diagnostic workload, but does
not yet solve end-to-end equality.

## Second controlled wave: 32629

Installed source inspection showed that compiled O3 defaults to
`custom_ops=['none']`. RMSNorm.forward_native bypasses the explicitly
batch-invariant branch in RMSNorm.forward_cuda. This is a candidate mechanism,
not a proven sole cause.

A 2 x 2 experiment varies the matrix tile size (stock vs small) and custom-op
selection (RMSNorm only vs all). Profile names:

- `invariant-o3-rms`: `custom_ops=['none','+rms_norm']`.
- `invariant-o3-all`: `custom_ops=['all']`.
- `invariant-smalltile-rms`: small BF16 matrix tiles, RMSNorm enabled.
- `invariant-smalltile-all`: small BF16 matrix tiles, all custom ops enabled.

Record actual GPU-worker precision flags and RMSNorm dispatch method, in
addition to parent configuration. Parent precision flags alone do not reveal
worker initialization. All changes are opt-in source-snapshot hooks; installed
vLLM files are unchanged. Raw outputs, actual runtime configs, GPU telemetry,
model checksums, source snapshots, comparisons and local backups are retained.

## Selection rule

A candidate must first match every diagnostic sequence and finish reason.
Then validate longer outputs and both target families before selecting it for
the full matched table. Record failed profiles and retained throughput; never
use diagnostic TPS as the final 128-request benchmark or mix runtime settings
between numerator and denominator.
