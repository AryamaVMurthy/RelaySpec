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

## Second-wave results

| Profile | AR TPS | Native DFlash TPS | Exact sequences / 8 |
|---|---:|---:|---:|
| Batch-invariant O3 + RMSNorm | 27.14 | 141.72 | 8 |
| Batch-invariant O3 + all custom ops | 27.05 | 138.47 | 8 |
| Smaller invariant tiles + RMSNorm | 41.10 | 196.48 | 8 |
| Smaller invariant tiles + all custom ops | 40.93 | 191.17 | 8 |

All four restored exact agreement on this diagnostic. Worker telemetry confirms
`RMSNorm.forward_cuda`, TF32 disabled, and reduced-precision BF16 reduction
disabled. The controlled RMSNorm ablation supports the compiler dispatch bypass
as a cause of the observed mismatch in this cohort. It does not establish
universal exactness across shapes, models or other versions.

Selected for longer validation: `invariant-smalltile-rms`. It retained 88.9%
of stock AR TPS and 93.5% of stock native TPS on the diagnostic. Between-profile
outputs differ; retained-TPS percentages are workload observations, not
identical-output latency comparisons. Within this profile AR and native have
identical 3,768 output tokens and a matched throughput ratio of 4.780x.

Job 32633: four paired AR/speculative pilots, eight requests each, cap 2,048:
Qwen native, Qwen single-matrix AUF, Llama native, and Qwen-to-Llama five-matrix
AUF. A non-exact comparison fails the pilot explicitly. This is still separate
from the final 128-request table.

Full confirmation is submitted as array 32637, dependent on all four 32633
pilots succeeding. Collector 32638 depends on that full array. Campaign:
`exact-nearpeak128-20260912`. The full run uses 128 prompts per target,
cap 2,048, four GPU-paired modulo shards and seven methods (fresh AR plus the
six existing speculative arms). Strict sequence/finish equality is an explicit
failure condition; collector also checks actual worker normalization dispatch.
Pending submission is not a completed measurement.
