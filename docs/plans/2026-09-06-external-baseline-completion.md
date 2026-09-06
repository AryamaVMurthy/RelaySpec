# External baseline completion after the small-data budget study

Large-data scaling remains paused. All further fit pools are512 or2048
distinct records. Keep the historical larger-data evidence and cache.
The full paper objective, including transfer, final confirmation and
bounded autoresearch after fixed baselines, remains active.

## Verified current evidence

- SD-square CPU setup27867 passed in1m20s with two CPUs and no GPU.
  Public source00f578aa and Transformers4.52.4 import under Python3.12.
  Its README specifies3.12 despite the pyproject declaring3.13.
  The native Qwen3-0.6B checkpoint is pinned and SHA-verified. The released
  steered checkpoint includes drafter tuning and is not relabeled frozen.
- Four-GPU pilot27869 finished in1m17s. Both public TVD and KL objectives
  trained only the402,665,472 steering parameters. Four updates visited
  four records from the declared512-example pool, not the whole pool.
  Training took1.24--1.35s. Duplicate weights, losses and gradients match
  exactly. Zero-guidance identity, frozen-drafter hashes, frozen-parameter
  version/storage checks and trainable/optimizer reload passed.
- The original exact-AR gate FAILED: both replicas of the second prompt
  first differ at output49. This remains a failed gate.
- Numerical replay27871 passed in1m01s. Same effective98-token prefix
  and logical position, with physical cache prefix129. Under SDPA, AR's
  logits favor59 over87 by36.25 versus36.0, while SD-square reverses that
  ordering. Eager attention reverses the choices again. Eager is not a
  solution to exact identity. The public AR helper also differs from
  correct cached AR on the measured first16-token output.
- Causal replay27877 passed in1m00s. Perturbing the six future query
  tokens preserves the first three query logits bit-for-bit under both
  SDPA and eager. Incoming cache copies are exact and independent, the
  original cache stays unchanged, and baseline replay logits match the
  actual verifier. This excludes future-query leakage in the tested call.
  It does not identify an exact CUDA kernel or prove a global guarantee.
- PARD's earlier cache/query replay27840 already establishes an analogous
  numerical sensitivity and causal intervention. Its exact-AR gate stays
  failed. PARD needs no new-target fitting.

Audited registries live in reports/external-baselines-20260906. Raw JSON
is collected under the original RelaySpec repository. Large checkpoint
and optimizer files remain on compute scratch. No long run is justified
merely by these compatibility results.

## Prospective checks and comparisons

The source-bound declaration is
configs/submission/baselines/external-verification-protocol.json.
It keeps every old exact-AR failure and separates three requirements:
faithful implementation, actual verifier decisions, and task quality.
Every committed token must equal the actual greedy verifier. Preserve
EOS and cap overshoot in raw records and charge complete request time.
Do not use the public SD-square AR helper as the reference. Use correct
cached AR with the same target weights, dtype, attention and token IDs.
Measure exact sequence agreement descriptively. Do not infer accuracy
from that agreement or describe a numerical mismatch as a quality pass.

1. Run the complete512-example SD-square fitting path inside a ten-minute
   pilot: batch4,128updates, learning rate1e-5 to1e-6, warmup8, two exact
   replicas each of TVD and KL. Exercise collation/loss masking, actual
   pool coverage, frozen weights, optimizer export/reload and decoding.
   Retain a separate compatibility result if numerical identity differs.
   A prospective pilot must pass its declared checks before any longer run.
2. If that pilot passes, compare4e-6 and2e-5 for each objective in a
   second four-GPU batch. Together with the pilot this gives six unique
   objective/rate settings. Account for all search cost. Keep public
   steering architecture and source statements unchanged.
3. Decode all six settings plus the initial unsteered drafter and
   runtime-local AR on the same16 exposed requests,256-token cap. Freeze
   rate/objective choice before larger quality data. If more training is
   warranted, use explicit warm-budget trajectories and preserved
   optimizer state, not a new large-data sweep.
4. Run PARD's prospective16-request development comparison with its
   pinned public runtime and matched correct AR. Its earlier exact-AR
   failure remains in the record. Keep inherited training separate from
   zero new-target fitting.
5. For selected external baselines, first run8 requests with2048-token
   caps, then the required128-request exposed quality comparison after
   the short path passes. Report quality intervals, cap counts, exact
   agreement and full request timing. No cross-runtime algorithm-only
   ranking: SD-square's public mixed precision, PARD's eager runtime and
   RelaySpec's runtime are different system configurations.
6. Only after fixed baseline and transfer evidence is complete, start
   the bounded autoresearch phase. Freeze final candidate/checkpoint,
   model/scorer revisions and comparison before untouched confirmation.
   Keep the existing one-percentage-point quality margin and95percent
   throughput-retention criterion. Inconclusive evidence stays inconclusive.

The work above completes a missing baseline track, not the entire paper.
The remaining small-data transfer/family, composition/complexity,
mechanism/serving, anonymous reproduction and final visual-review work
must still be checked against the original execution plan.

## Current execution checkpoint

Full-pool pilot27888 and rate screen27890 passed in2m04s each. Every fit
visited512 examples in128 batch4 updates. Training took47--48seconds.
The first common decoding job27895 was cancelled at3m47s because its
drafter and steering retained FP32 training storage. Preserve partial
artifacts and exclude its timings. The corrected campaign follows public
eval.py with BF16 drafter/steering and FP16 inherited target, under BF16
autocast. Verify original FP32 training fingerprints before conversion
and separately fingerprint the actual BF16 inference tensors.

Corrected inference job27898 was cancelled at5m13s after measured
SD-square requests took10--15seconds for256 tokens. The full nine-method
sixteen-request batch would exceed the pilot limit once setup and
verification are included. Preserve partial artifacts without a completion
claim. Execute the same fixed comparison in two disjoint eight-request
shards, each with four GPUs and a ten-minute allocation. Aggregate only
after both source-bound gates pass and the sixteen prompts are disjoint.

Launched SD-square shards27900/27901 from3e96e78. The second depends on
the first passing. PARD campaign27903 frome3bb73e depends on the second
shard and uses the same sixteen development prompts. All jobs request
four GPUs with ten-minute limits. PARD verifies a separately observed
replica of every timed request against its actual target argmax decisions
and checks raw tokens and acceptance against the trace-free timed run.
Public generation statements remain unchanged, tested by AST equality.
Auditors rebuild source-bound gates before creating result registries.
